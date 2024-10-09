use std::{
    io::{Read, Seek},
    path::{Path, PathBuf},
};

fn get_data_dir() -> PathBuf {
    let this_file = PathBuf::from(file!());
    let this_dir = this_file.parent().unwrap();
    this_dir.join("data")
}

fn get_scripts() -> Vec<PathBuf> {
    let data_dir = get_data_dir();
    data_dir
        .read_dir()
        .unwrap()
        .map(|path| path.unwrap().path())
        .filter(|path| path.is_file() && path.ends_with(".lox"))
        .collect::<Vec<PathBuf>>()
}

fn extract_extracted_output(file_path: &Path) -> String {
    let script_content = std::fs::read_to_string(file_path).unwrap();
    let output_lines = script_content
        .split('\n')
        .take_while(|line| line.starts_with("//"))
        .map(|line| line.replace("//", ""))
        .collect::<Vec<String>>();
    textwrap::dedent(&output_lines.join("\n"))
}

fn execute_script(file_path: &Path) -> String {
    let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
    lox::execute(&file_path, &mut cursor).unwrap();

    cursor.seek(std::io::SeekFrom::Start(0)).unwrap();
    let mut string_output = String::new();
    let _ = cursor.read_to_string(&mut string_output);
    String::from(string_output.trim_end())
}

/// Test function for Script files.
///
/// Each script file contains the code to execute as well as the expected output.
/// The structure of a script file is as follows:
///
/// ```text
///   // expected-output-line1
///   // ...
///   // expected-output-lineN
///   Lox code
/// ```
#[test]
fn test_scripts() {
    let scripts = get_scripts();

    for file_path in scripts {
        let expected_output = extract_extracted_output(&file_path);
        let captured_output = execute_script(&file_path);

        assert_eq!(captured_output, expected_output);
    }
}
