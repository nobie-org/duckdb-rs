use crate::{error::Error, inner_connection::InnerConnection, Connection, Result};

use super::{ffi, ffi::duckdb_free};
use std::ffi::c_void;

mod function;
mod value;

/// The duckdb Arrow table function interface
#[cfg(feature = "vtab-arrow")]
pub mod arrow;
#[cfg(feature = "vtab-arrow")]
pub use self::arrow::{
    arrow_arraydata_to_query_params, arrow_ffi_to_query_params, arrow_recordbatch_to_query_params,
    record_batch_to_duckdb_data_chunk, to_duckdb_logical_type, to_duckdb_type_id,
};
#[cfg(feature = "vtab-excel")]
mod excel;

use function::ScalarFunction;
pub use function::{BindInfo, FunctionInfo, InitInfo, TableFunction};
use libduckdb_sys::duckdb_vector;
pub use value::Value;

use crate::core::{DataChunkHandle, FlatVector, LogicalTypeHandle, LogicalTypeId};
use ffi::{duckdb_bind_info, duckdb_data_chunk, duckdb_function_info, duckdb_init_info};

use ffi::duckdb_malloc;
use std::mem::size_of;

/// duckdb_malloc a struct of type T
/// used for the bind_info and init_info
/// # Safety
/// This function is obviously unsafe
unsafe fn malloc_data_c<T>() -> *mut T {
    duckdb_malloc(size_of::<T>()).cast()
}

/// free bind or info data
///
/// # Safety
///   This function is obviously unsafe
/// TODO: maybe we should use a Free trait here
unsafe extern "C" fn drop_data_c<T: Free>(v: *mut c_void) {
    let actual = v.cast::<T>();
    (*actual).free();
    duckdb_free(v);
}

/// Free trait for the bind and init data
pub trait Free {
    /// Free the data
    fn free(&mut self) {}
}

/// Duckdb table function trait
///
/// See to the HelloVTab example for more details
/// <https://duckdb.org/docs/api/c/table_functions>
pub trait VTab: Sized {
    /// The data type of the bind data
    type InitData: Sized + Free;
    /// The data type of the init data
    type BindData: Sized + Free;

    /// Bind data to the table function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers (`data`) and manipulates the memory directly.
    /// The caller must ensure that:
    ///
    /// - The `data` pointer is valid and points to a properly initialized `BindData` instance.
    /// - The lifetime of `data` must outlive the execution of `bind` to avoid dangling pointers, especially since
    ///   `bind` does not take ownership of `data`.
    /// - Concurrent access to `data` (if applicable) must be properly synchronized.
    /// - The `bind` object must be valid and correctly initialized.
    unsafe fn bind(bind: &BindInfo, data: *mut Self::BindData) -> Result<(), Box<dyn std::error::Error>>;
    /// Initialize the table function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it performs raw pointer dereferencing on the `data` argument.
    /// The caller is responsible for ensuring that:
    ///
    /// - The `data` pointer is non-null and points to a valid `InitData` instance.
    /// - There is no data race when accessing `data`, meaning if `data` is accessed from multiple threads,
    ///   proper synchronization is required.
    /// - The lifetime of `data` extends beyond the scope of this call to avoid use-after-free errors.
    unsafe fn init(init: &InitInfo, data: *mut Self::InitData) -> Result<(), Box<dyn std::error::Error>>;
    /// The actual function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    ///
    /// - Dereferences multiple raw pointers (`func` to access `init_info` and `bind_info`).
    ///
    /// The caller must ensure that:
    ///
    /// - All pointers (`func`, `output`, internal `init_info`, and `bind_info`) are valid and point to the expected types of data structures.
    /// - The `init_info` and `bind_info` data pointed to remains valid and is not freed until after this function completes.
    /// - No other threads are concurrently mutating the data pointed to by `init_info` and `bind_info` without proper synchronization.
    /// - The `output` parameter is correctly initialized and can safely be written to.
    unsafe fn func(func: &FunctionInfo, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>>;
    /// Does the table function support pushdown
    /// default is false
    fn supports_pushdown() -> bool {
        false
    }
    /// The parameters of the table function
    /// default is None
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }
    /// The named parameters of the table function
    /// default is None
    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        None
    }
}

unsafe extern "C" fn func<T>(info: duckdb_function_info, output: duckdb_data_chunk)
where
    T: VTab,
{
    let info = FunctionInfo::from(info);
    let mut data_chunk_handle = DataChunkHandle::new_unowned(output);
    let result = T::func(&info, &mut data_chunk_handle);
    if result.is_err() {
        info.set_error(&result.err().unwrap().to_string());
    }
}

unsafe extern "C" fn init<T>(info: duckdb_init_info)
where
    T: VTab,
{
    let info = InitInfo::from(info);
    let data = malloc_data_c::<T::InitData>();
    let result = T::init(&info, data);
    info.set_init_data(data.cast(), Some(drop_data_c::<T::InitData>));
    if result.is_err() {
        info.set_error(&result.err().unwrap().to_string());
    }
}

unsafe extern "C" fn bind<T>(info: duckdb_bind_info)
where
    T: VTab,
{
    let info = BindInfo::from(info);
    let data = malloc_data_c::<T::BindData>();
    let result = T::bind(&info, data);
    info.set_bind_data(data.cast(), Some(drop_data_c::<T::BindData>));
    if result.is_err() {
        info.set_error(&result.err().unwrap().to_string());
    }
}

/// Duckdb scalar function trait
///
pub trait VScalar: Sized {
    // type ExtraInfo;
    /// The actual function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    ///
    /// - Dereferences multiple raw pointers (`func``).
    ///
    unsafe fn func(
        func: &FunctionInfo,
        input: &mut DataChunkHandle,
        output: &mut FlatVector,
    ) -> Result<(), Box<dyn std::error::Error>>;
    /// The parameters of the table function
    /// default is None
    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }

    /// The return type of the scalar function
    /// default is None
    fn return_type() -> LogicalTypeHandle;
}

unsafe extern "C" fn scalar_func<T>(info: duckdb_function_info, input: duckdb_data_chunk, output: duckdb_vector)
where
    T: VScalar,
{
    let info = FunctionInfo::from(info);
    let mut input = DataChunkHandle::new_unowned(input);
    let mut output_vector = FlatVector::from(output);
    let result = T::func(&info, &mut input, &mut output_vector);
    if result.is_err() {
        info.set_error(&result.err().unwrap().to_string());
    }
}

fn drop_type<T>(v: *mut c_void) {
    let actual = v.cast::<T>();
    unsafe {
        Box::from_raw(actual);
    }
}

impl Connection {
    /// Register the given TableFunction with the current db
    #[inline]
    pub fn register_table_function<T: VTab>(&self, name: &str) -> Result<()> {
        let table_function = TableFunction::default();
        table_function
            .set_name(name)
            .supports_pushdown(T::supports_pushdown())
            .set_bind(Some(bind::<T>))
            .set_init(Some(init::<T>))
            .set_function(Some(func::<T>));
        for ty in T::parameters().unwrap_or_default() {
            table_function.add_parameter(&ty);
        }
        for (name, ty) in T::named_parameters().unwrap_or_default() {
            table_function.add_named_parameter(&name, &ty);
        }
        self.db.borrow_mut().register_table_function(table_function)
    }

    /// Register the given ScalarFunction with the current db
    #[inline]
    pub fn register_scalar_function<S: VScalar>(&self, name: &str) -> Result<()> {
        let scalar_function = ScalarFunction::default();
        scalar_function
            .set_name(name)
            .set_return_type(&S::return_type())
            // .set_extra_info()
            .set_function(Some(scalar_func::<S>));
        for ty in S::parameters().unwrap_or_default() {
            scalar_function.add_parameter(&ty);
        }
        self.db.borrow_mut().register_scalar_function(scalar_function)
    }
}

impl InnerConnection {
    /// Register the given TableFunction with the current db
    pub fn register_table_function(&mut self, table_function: TableFunction) -> Result<()> {
        unsafe {
            let rc = ffi::duckdb_register_table_function(self.con, table_function.ptr);
            if rc != ffi::DuckDBSuccess {
                return Err(Error::DuckDBFailure(ffi::Error::new(rc), None));
            }
        }
        Ok(())
    }

    /// Register the given ScalarFunction with the current db
    pub fn register_scalar_function(&mut self, scalar_function: ScalarFunction) -> Result<()> {
        unsafe {
            let rc = ffi::duckdb_register_scalar_function(self.con, scalar_function.ptr);
            if rc != ffi::DuckDBSuccess {
                return Err(Error::DuckDBFailure(ffi::Error::new(rc), None));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::Inserter;
    use core::panic;
    use std::{
        borrow::Cow,
        error::Error,
        ffi::{c_char, CStr, CString},
        marker::PhantomData,
    };

    #[repr(C)]
    struct HelloBindData {
        name: *mut c_char,
    }

    impl Free for HelloBindData {
        fn free(&mut self) {
            unsafe {
                if self.name.is_null() {
                    return;
                }
                drop(CString::from_raw(self.name));
            }
        }
    }

    #[repr(C)]
    struct HelloInitData {
        done: bool,
    }

    struct HelloVTab;

    impl Free for HelloInitData {}

    impl VTab for HelloVTab {
        type InitData = HelloInitData;
        type BindData = HelloBindData;

        unsafe fn bind(bind: &BindInfo, data: *mut HelloBindData) -> Result<(), Box<dyn std::error::Error>> {
            bind.add_result_column("column0", LogicalTypeHandle::from(LogicalTypeId::Varchar));
            let param = bind.get_parameter(0).to_string();
            unsafe {
                (*data).name = CString::new(param).unwrap().into_raw();
            }
            Ok(())
        }

        unsafe fn init(_: &InitInfo, data: *mut HelloInitData) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                (*data).done = false;
            }
            Ok(())
        }

        unsafe fn func(func: &FunctionInfo, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
            let init_info = func.get_init_data::<HelloInitData>();
            let bind_info = func.get_bind_data::<HelloBindData>();

            unsafe {
                if (*init_info).done {
                    output.set_len(0);
                } else {
                    (*init_info).done = true;
                    let vector = output.flat_vector(0);
                    let name = CString::from_raw((*bind_info).name);
                    let result = CString::new(format!("Hello {}", name.to_str()?))?;
                    // Can't consume the CString
                    (*bind_info).name = CString::into_raw(name);
                    vector.insert(0, result);
                    output.set_len(1);
                }
            }
            Ok(())
        }

        fn parameters() -> Option<Vec<LogicalTypeHandle>> {
            Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
        }
    }

    struct HelloWithNamedVTab {}
    impl VTab for HelloWithNamedVTab {
        type InitData = HelloInitData;
        type BindData = HelloBindData;

        unsafe fn bind(bind: &BindInfo, data: *mut HelloBindData) -> Result<(), Box<dyn Error>> {
            bind.add_result_column("column0", LogicalTypeHandle::from(LogicalTypeId::Varchar));
            let param = bind.get_named_parameter("name").unwrap().to_string();
            assert!(bind.get_named_parameter("unknown_name").is_none());
            unsafe {
                (*data).name = CString::new(param).unwrap().into_raw();
            }
            Ok(())
        }

        unsafe fn init(init_info: &InitInfo, data: *mut HelloInitData) -> Result<(), Box<dyn Error>> {
            HelloVTab::init(init_info, data)
        }

        unsafe fn func(func: &FunctionInfo, output: &mut DataChunkHandle) -> Result<(), Box<dyn Error>> {
            HelloVTab::func(func, output)
        }

        fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
            Some(vec![(
                "name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            )])
        }
    }

    struct HelloTestFunction {}

    impl VScalar for HelloTestFunction {
        unsafe fn func(
            func: &FunctionInfo,
            input: &mut DataChunkHandle,
            output: &mut FlatVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            println!("Function info {:?}", func);
            let rows = input.len();
            for _ in 0..rows {
                output.insert(0, "hello");
            }
            Ok(())
        }

        fn return_type() -> LogicalTypeHandle {
            LogicalTypeHandle::from(LogicalTypeId::Varchar)
        }
    }

    // Add a lifetime parameter and PhantomData to tie it to that lifetime
    struct DuckString<'a> {
        ptr: *mut duckdb_string_t,
        _phantom: PhantomData<&'a mut duckdb_string_t>,
    }

    impl<'a> DuckString<'a> {
        fn new(ptr: *mut duckdb_string_t) -> Self {
            DuckString {
                ptr,
                _phantom: PhantomData,
            }
        }
    }

    impl<'a> DuckString<'a> {
        fn as_str(&self) -> std::borrow::Cow<'a, str> {
            unsafe { CStr::from_ptr(duckdb_string_t_data(self.ptr)).to_string_lossy() }
        }
    }

    struct Repeat {}

    impl VScalar for Repeat {
        unsafe fn func(
            _func: &FunctionInfo,
            input: &mut DataChunkHandle,
            output: &mut FlatVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let flat_counts = input.flat_vector(1);
            let values = input.flat_vector(0);
            let values = values.as_slice::<*mut duckdb_string_t>();
            let strings = values.iter().map(|ptr| DuckString::new(*ptr).as_str());
            let counts = flat_counts.as_slice::<i32>();
            for (count, value) in counts.iter().zip(strings) {
                output.insert(0, value.repeat((*count) as usize).as_str());
            }

            Ok(())
        }

        fn return_type() -> LogicalTypeHandle {
            LogicalTypeHandle::from(LogicalTypeId::Varchar)
        }

        fn parameters() -> Option<Vec<LogicalTypeHandle>> {
            Some(vec![
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
                LogicalTypeHandle::from(LogicalTypeId::Integer),
            ])
        }
    }

    #[test]
    fn test_table_function() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_table_function::<HelloVTab>("hello")?;

        let val = conn.query_row("select * from hello('duckdb')", [], |row| <(String,)>::try_from(row))?;
        assert_eq!(val, ("Hello duckdb".to_string(),));

        Ok(())
    }

    #[test]
    fn test_named_table_function() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_table_function::<HelloWithNamedVTab>("hello_named")?;

        let val = conn.query_row("select * from hello_named(name = 'duckdb')", [], |row| {
            <(String,)>::try_from(row)
        })?;
        assert_eq!(val, ("Hello duckdb".to_string(),));

        Ok(())
    }

    #[test]
    fn test_scalar_function() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_scalar_function::<HelloTestFunction>("hello")?;

        let val = conn.query_row("select hello() as hello", [], |row| <(String,)>::try_from(row))?;
        assert_eq!(val, ("hello".to_string(),));

        let batches = conn
            .prepare("select hello() as hello from range(10)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    #[test]
    fn test_repeat() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_scalar_function::<Repeat>("nobie_repeat")?;

        let batches = conn
            .prepare("select repeat('Ho ho ho, 🎅🎄', 10) as message from range(8)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    use ::arrow::util::pretty::print_batches;
    #[cfg(feature = "vtab-loadable")]
    use duckdb_loadable_macros::duckdb_entrypoint;
    use libduckdb_sys::{duckdb_result_arrow_array, duckdb_string_t, duckdb_string_t_data};

    // this function is never called, but is still type checked
    // Exposes a extern C function named "libhello_ext_init" in the compiled dynamic library,
    // the "entrypoint" that duckdb will use to load the extension.
    #[cfg(feature = "vtab-loadable")]
    #[duckdb_entrypoint]
    fn libhello_ext_init(conn: Connection) -> Result<(), Box<dyn Error>> {
        conn.register_table_function::<HelloVTab>("hello")?;
        Ok(())
    }
}
