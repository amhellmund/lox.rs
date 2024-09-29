// Copyright (c) 2024 Andi Hellmund. All rights reserved.

// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

use std::error;
use std::fmt;
use std::path::PathBuf;

/// Location in the code file.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Location {
    pub line: i64,
    pub column: i64,
}

impl Location {
    pub fn new(line: i64, column: i64) -> Self {
        Location { line, column }
    }
}

impl ToString for Location {
    fn to_string(&self) -> String {
        format!("{}:{}", self.line, self.column)
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct LocationSpan {
    pub start: Location,
    pub end_inclusive: Location,
}

impl ToString for LocationSpan {
    fn to_string(&self) -> String {
        format!(
            "{}:{}-{}:{}",
            self.start.line, self.start.column, self.end_inclusive.line, self.end_inclusive.column
        )
    }
}

impl LocationSpan {
    pub fn new(start: Location, end_inclusive: Location) -> Self {
        LocationSpan {
            start,
            end_inclusive,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FileLocation {
    SinglePoint(Location),
    Span(LocationSpan),
}

impl ToString for FileLocation {
    fn to_string(&self) -> String {
        match *self {
            FileLocation::SinglePoint(point) => point.to_string(),
            FileLocation::Span(span) => span.to_string(),
        }
    }
}

#[derive(Debug)]
pub struct DiagnosticError {
    pub message: String,
    pub location: FileLocation,
    pub source_file: PathBuf,
}

impl DiagnosticError {
    pub fn new(message: String, location: FileLocation, source_file: PathBuf) -> Self {
        DiagnosticError {
            message,
            location,
            source_file,
        }
    }
}

impl DiagnosticError {
    fn get_source_file_name(&self) -> &str {
        self.source_file.file_name().unwrap().to_str().unwrap()
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.location {
            FileLocation::SinglePoint(point) => {
                write!(
                    f,
                    "{} [{}@{}] ",
                    self.message,
                    self.get_source_file_name(),
                    point.to_string(),
                )
            }
            FileLocation::Span(span) => {
                write!(
                    f,
                    "{} [{}@{}] ",
                    self.message,
                    self.get_source_file_name(),
                    span.to_string(),
                )
            }
        }
    }
}

impl error::Error for DiagnosticError {}
