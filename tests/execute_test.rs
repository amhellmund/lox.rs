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
        .filter(|path| path.is_file())
        .collect::<Vec<PathBuf>>()
}

fn extract_extracted_output(file_path: &Path) -> String {
    String::default()
}

#[test]
fn test_scripts() {
    let scripts = get_scripts();

    for script in scripts {
        let expected_output = extract_extracted_output(&script);

        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        lox::execute(&script, &mut cursor).unwrap();

        cursor.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut string_output = String::new();
        let _ = cursor.read_to_string(&mut string_output);
        let captured_output = String::from(string_output.trim_end());

        assert_eq!(captured_output, expected_output);
    }
}
