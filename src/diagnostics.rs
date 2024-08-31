use std::fmt;
use std::error;
use std::path::PathBuf;

/// Location in the code file
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Location {
    pub line: i64,
    pub column: i64,
}

#[derive(Debug)]
pub struct DiagnosticError {
    pub message: String,
    pub location: Location,
    pub source_file: PathBuf,
}

impl DiagnosticError {
    pub fn new (message: String, location: Location, source_file: PathBuf) -> Self {
        DiagnosticError{
            message,
            location,
            source_file
        }
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}@{}:{}] ", self.message, self.source_file.display(), self.location.line, self.location.column)
    }
}

impl error::Error for DiagnosticError {
}