use std::fmt::Display;

use nom::error::VerboseError;

#[derive(Debug)]
pub enum Error {
    ParserError(String),
}

impl<'a> Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ParserError(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl<'a> From<VerboseError<&'a str>> for Error {
    fn from(from: VerboseError<&'a str>) -> Self {
        Self::ParserError(from.to_string())
    }
}
