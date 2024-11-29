use crate::Arrow;
use crate::{error::Error, inner_connection::InnerConnection, Connection, Result};

use super::{ffi, ffi::duckdb_free};
use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

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

use ::arrow::array::{Array, RecordBatch};
use ::arrow::datatypes::DataType;
use arrow::{data_chunk_to_arrow, write_arrow_array_to_vector, WritableVector};
pub use function::{BindInfo, FunctionInfo, InitInfo, TableFunction};
use function::{ScalarFunction, ScalarFunctionSet};
use libduckdb_sys::{duckdb_string_t, duckdb_vector};
pub use value::Value;

use crate::core::{DataChunkHandle, FlatVector, LogicalTypeHandle, LogicalTypeId, Vector};
use ffi::{duckdb_bind_info, duckdb_data_chunk, duckdb_function_info, duckdb_init_info};

use ffi::duckdb_malloc;
use std::mem::size_of;

/// duckdb_malloc a struct of type T
/// used for the bind_info and init_info
/// # Safety
/// This function is obviously unsafe
pub unsafe fn malloc_data_c<T>() -> *mut T {
    duckdb_malloc(size_of::<T>()).cast()
}

/// free bind or info data
///
/// # Safety
///   This function is obviously unsafe
/// TODO: maybe we should use a Free trait here
pub unsafe extern "C" fn drop_data_c<T: Free>(v: *mut c_void) {
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

struct ScalarFunctionSignature {
    parameters: Option<Vec<LogicalTypeHandle>>,
    return_type: LogicalTypeHandle,
}

impl ScalarFunctionSignature {
    fn register_with_scalar(&self, f: &ScalarFunction) {
        f.set_return_type(&self.return_type);

        for param in self.parameters.iter().flatten() {
            f.add_parameter(param);
        }
    }
}

/// Duckdb scalar function trait
///
pub trait VScalar: Sized {
    /// blah
    type Info: Default;
    /// The actual function
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    ///
    /// - Dereferences multiple raw pointers (`func``).
    ///
    unsafe fn func(
        func: &Self::Info,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// The possible signatures of the scalar function
    fn signatures() -> Vec<ScalarFunctionSignature>;

    // /// The parameters of the table function
    // /// default is None
    // fn parameters() -> Option<Vec<LogicalTypeHandle>> {
    //     None
    // }

    // /// The return type of the scalar function
    // /// default is None
    // fn return_type() -> LogicalTypeHandle;
}

struct ArrowFunctionSignature {
    pub parameters: Option<Vec<DataType>>,
    pub return_type: DataType,
}

/// blah
pub trait ArrowScalar: Sized {
    /// blah
    type FuncInfo: Default;

    /// blah
    fn func(info: &Self::FuncInfo, input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn std::error::Error>>;

    fn signatures() -> Vec<ArrowFunctionSignature>;
}

impl<T> VScalar for T
where
    T: ArrowScalar,
    T::FuncInfo: Debug,
{
    type Info = T::FuncInfo;

    unsafe fn func(
        info: &Self::Info,
        input: &mut DataChunkHandle,
        out: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("info: {:?}", info);
        let array = T::func(info, data_chunk_to_arrow(input)?)?;
        write_arrow_array_to_vector(&array, out)
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        T::signatures()
            .into_iter()
            .map(|sig| ScalarFunctionSignature {
                parameters: sig.parameters.map(|v| {
                    v.into_iter()
                        .flat_map(|dt| LogicalTypeId::try_from(&dt).ok().map(Into::into))
                        .collect()
                }),
                return_type: LogicalTypeId::try_from(&sig.return_type)
                    .ok()
                    .map(Into::into)
                    .unwrap_or_else(|| LogicalTypeHandle::from(LogicalTypeId::Integer)),
            })
            .collect()
    }
}

/// blah
pub trait VScalarFlatVector: Sized {
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

unsafe extern "C" fn scalar_func<T>(info: duckdb_function_info, input: duckdb_data_chunk, mut output: duckdb_vector)
where
    T: VScalar,
{
    let info = FunctionInfo::from(info);
    let mut input = DataChunkHandle::new_unowned(input);
    let result = T::func(info.get_scalar_extra_info(), &mut input, &mut output);
    if result.is_err() {
        info.set_error(&result.err().unwrap().to_string());
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
    pub fn register_scalar_function<S: VScalar>(&self, name: &str) -> Result<()>
    where
        S::Info: Debug,
    {
        let set = ScalarFunctionSet::new(name);
        for signature in S::signatures() {
            let scalar_function = ScalarFunction::new(name)?;
            signature.register_with_scalar(&scalar_function);
            scalar_function.set_function(Some(scalar_func::<S>));
            scalar_function.set_extra_info::<S::Info>();
            set.add_function(scalar_function)?;
        }
        self.db.borrow_mut().register_scalar_function_set(set)
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
    pub fn register_scalar_function_set(&mut self, f: ScalarFunctionSet) -> Result<()> {
        f.register_with_connection(self.con)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::Inserter;
    use std::{
        error::Error,
        ffi::{c_char, CString},
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

    struct HelloArrowScalar {}

    impl ArrowScalar for HelloArrowScalar {
        type FuncInfo = ();

        fn func(info: &Self::FuncInfo, input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn std::error::Error>> {
            let name = input.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let result = name.iter().map(|v| format!("Hello {}", v.unwrap())).collect::<Vec<_>>();
            Ok(Arc::new(StringArray::from(result)))
        }

        fn signatures() -> Vec<ArrowFunctionSignature> {
            vec![ArrowFunctionSignature {
                parameters: Some(vec![DataType::Utf8]),
                return_type: DataType::Utf8,
            }]
        }
    }

    #[derive(Debug)]
    struct MockMeta {
        name: String,
    }

    impl Default for MockMeta {
        fn default() -> Self {
            MockMeta {
                name: "some meta".to_string(),
            }
        }
    }

    impl Drop for MockMeta {
        fn drop(&mut self) {
            println!("dropped meta");
        }
    }

    struct ArrowMultiplyScalar {}

    impl ArrowScalar for ArrowMultiplyScalar {
        type FuncInfo = MockMeta;

        fn func(info: &Self::FuncInfo, input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn std::error::Error>> {
            println!("info: {:?}", info);

            let a = input
                .column(0)
                .as_any()
                .downcast_ref::<::arrow::array::Float32Array>()
                .unwrap();

            let b = input
                .column(1)
                .as_any()
                .downcast_ref::<::arrow::array::Float32Array>()
                .unwrap();

            let result = a
                .iter()
                .zip(b.iter())
                .map(|(a, b)| a.unwrap() * b.unwrap())
                .collect::<Vec<_>>();
            Ok(Arc::new(::arrow::array::Float32Array::from(result)))
        }

        fn signatures() -> Vec<ArrowFunctionSignature> {
            vec![ArrowFunctionSignature {
                parameters: Some(vec![DataType::Float32, DataType::Float32]),
                return_type: DataType::Float32,
            }]
        }
    }

    // accepts a string or a number and parses to int and multiplies by 2
    struct ArrowMultipleSignatureScalar {}

    impl ArrowScalar for ArrowMultipleSignatureScalar {
        type FuncInfo = MockMeta;

        fn func(info: &Self::FuncInfo, input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn std::error::Error>> {
            println!("info: {:?}", info);

            let a = input.column(0);
            let b = input.column(1);

            let result = match a.data_type() {
                DataType::Utf8 => {
                    let a = a
                        .as_any()
                        .downcast_ref::<::arrow::array::StringArray>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap().parse::<f32>().unwrap())
                        .collect::<Vec<_>>();
                    let b = b
                        .as_any()
                        .downcast_ref::<::arrow::array::Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap())
                        .collect::<Vec<_>>();
                    a.iter().zip(b.iter()).map(|(a, b)| a * b).collect::<Vec<_>>()
                }
                DataType::Float32 => {
                    let a = a
                        .as_any()
                        .downcast_ref::<::arrow::array::Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap())
                        .collect::<Vec<_>>();
                    let b = b
                        .as_any()
                        .downcast_ref::<::arrow::array::Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap())
                        .collect::<Vec<_>>();
                    a.iter().zip(b.iter()).map(|(a, b)| a * b).collect::<Vec<_>>()
                }
                _ => panic!("unsupported type"),
            };

            Ok(Arc::new(::arrow::array::Float32Array::from(result)))
        }

        fn signatures() -> Vec<ArrowFunctionSignature> {
            vec![
                ArrowFunctionSignature {
                    parameters: Some(vec![DataType::Utf8, DataType::Float32]),
                    return_type: DataType::Float32,
                },
                ArrowFunctionSignature {
                    parameters: Some(vec![DataType::Float32, DataType::Float32]),
                    return_type: DataType::Float32,
                },
            ]
        }
    }

    struct HelloTestFunction {}

    impl VScalar for HelloTestFunction {
        type Info = ();

        unsafe fn func(
            _func: &Self::Info,
            input: &mut DataChunkHandle,
            output: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let values = input.flat_vector(0);
            let values = values.as_slice_with_len::<duckdb_string_t>(input.len());
            let strings = values
                .iter()
                .map(|ptr| arrow::DuckString::new(&mut { *ptr }).as_str().to_string())
                .take(input.len());
            let output = output.flat_vector();
            for s in strings {
                output.insert(0, s.to_string().as_str());
            }
            Ok(())
        }

        fn signatures() -> Vec<ScalarFunctionSignature> {
            vec![ScalarFunctionSignature {
                parameters: Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)]),
                return_type: LogicalTypeHandle::from(LogicalTypeId::Varchar),
            }]
        }
    }

    struct Repeat {}

    impl VScalar for Repeat {
        type Info = ();

        unsafe fn func(
            _func: &Self::Info,
            input: &mut DataChunkHandle,
            output: &mut dyn WritableVector,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let output = output.flat_vector();
            let counts = input.flat_vector(1);
            let values = input.flat_vector(0);
            let values = values.as_slice_with_len::<duckdb_string_t>(input.len());
            let strings = values
                .iter()
                .map(|ptr| arrow::DuckString::new(&mut { *ptr }).as_str().to_string());
            let counts = counts.as_slice_with_len::<i32>(input.len());
            for (count, value) in counts.iter().zip(strings).take(input.len()) {
                output.insert(0, value.repeat((*count) as usize).as_str());
            }

            Ok(())
        }

        fn signatures() -> Vec<ScalarFunctionSignature> {
            vec![ScalarFunctionSignature {
                parameters: Some(vec![
                    LogicalTypeHandle::from(LogicalTypeId::Varchar),
                    LogicalTypeHandle::from(LogicalTypeId::Integer),
                ]),
                return_type: LogicalTypeHandle::from(LogicalTypeId::Varchar),
            }]
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

        // let val = conn.query_row("select hello('matt') as hello", [], |row| <(String,)>::try_from(row))?;
        // assert_eq!(val, ("hello".to_string(),));

        let batches = conn
            .prepare("select hello('matt') as hello from range(10)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    #[test]
    fn test_arrow_scalar_function() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_scalar_function::<HelloArrowScalar>("hello")?;

        // let val = conn.query_row("select hello('matt') as hello", [], |row| <(String,)>::try_from(row))?;
        // assert_eq!(val, ("hello".to_string(),));

        let batches = conn
            .prepare("select hello('matt') as hello from range(10)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    #[test]
    fn test_arrow_scalar_multiply_function() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_scalar_function::<ArrowMultiplyScalar>("nobie_multiply")?;

        let batches = conn
            .prepare("select nobie_multiply(3.0, 2.0) as mult_result from range(10)")?
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
            .prepare("select nobie_repeat('Ho ho ho, 🎅🎄', 10) as message from range(8)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    #[test]
    fn test_multiple_signatures() -> Result<(), Box<dyn Error>> {
        let conn = Connection::open_in_memory()?;
        conn.register_scalar_function::<ArrowMultipleSignatureScalar>("nobie_multi_sig")?;

        let batches = conn
            .prepare("select nobie_multi_sig('1230.0', 5) as message from range(8)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        let batches = conn
            .prepare("select nobie_multi_sig(12, 10) as message from range(8)")?
            .query_arrow([])?
            .collect::<Vec<_>>();

        print_batches(&batches)?;

        Ok(())
    }

    use ::arrow::{array::StringArray, util::pretty::print_batches};
    #[cfg(feature = "vtab-loadable")]
    use duckdb_loadable_macros::duckdb_entrypoint;
    use libduckdb_sys::{duckdb_result_arrow_array, duckdb_string_t, duckdb_string_t_data};
    use num::Float;

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
