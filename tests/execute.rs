use pretty_assertions::assert_eq;
use std::{
    ffi::OsStr,
    io::{Read, Seek},
    path::{Path, PathBuf},
};

fn get_data_dir() -> PathBuf {
    let this_file = PathBuf::from(file!());
    dbg!(&this_file);
    let this_dir = this_file.parent().unwrap();
    this_dir.join("data")
}

fn get_scripts() -> Vec<PathBuf> {
    let data_dir = get_data_dir().join("scripts");
    let mut scripts = data_dir
        .read_dir()
        .unwrap()
        .map(|path| path.unwrap().path())
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("lox")))
        .collect::<Vec<PathBuf>>();
    scripts.sort();
    scripts
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

fn get_fixture_file() -> PathBuf {
    get_data_dir().join("fixture.txt")
}

fn read_fixture() -> Vec<PathBuf> {
    let content = std::fs::read_to_string(get_fixture_file()).unwrap();
    content
        .split('\n')
        .map(|line| PathBuf::from(line.trim()))
        .collect::<Vec<_>>()
}

fn write_fixture(content: &Vec<PathBuf>) {
    let mut content = content.clone();
    content.sort();
    std::fs::write(
        get_fixture_file(),
        content
            .iter()
            .map(|file| String::from(file.to_str().unwrap()))
            .collect::<Vec<String>>()
            .join("\n"),
    )
    .unwrap();
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
    let scripts = &get_scripts();

    for file_path in scripts {
        let expected_output = extract_extracted_output(&file_path);
        let captured_output = execute_script(&file_path);

        assert_eq!(captured_output, expected_output);
    }

    let current_fixture = read_fixture();
    write_fixture(scripts);
    assert_eq!(&current_fixture, scripts);
}
