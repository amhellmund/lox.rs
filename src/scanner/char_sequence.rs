use crate::diagnostics::Location;

/// Wrapper around a sequence of characters providing convenience functions:
///
///   o Look at characters from the current position in the sequence
///   o Advance position while updating the location information
///   o End-of-sequence check
pub struct CharSequence {
    chars: Vec<char>,
    pos: usize,
    location: Location,
}

impl CharSequence {
    pub fn new(chars: &str) -> Self {
        Self {
            chars: chars.chars().collect(),
            pos: 0,
            location: Location::new(1, 1),
        }
    }

    pub fn look_at(&self, num: usize) -> Option<char> {
        let index = self.pos + num;
        if index < self.chars.len() {
            Some(self.chars[index])
        } else {
            None
        }
    }

    pub fn advance(&mut self, num: usize) {
        for _ in 0..num {
            if self.pos < self.chars.len() {
                let ch = self.chars[self.pos];
                if ch == '\n' {
                    self.location.line += 1;
                    self.location.column = 1;
                } else if ch != '\r' {
                    self.location.column += 1;
                }
                self.pos += 1;
            }
        }
    }

    pub fn has_reached_end(&self) -> bool {
        return self.pos >= self.chars.len();
    }

    pub fn location(&self) -> Location {
        self.location
    }
}

#[cfg(test)]
mod tests {
    use super::CharSequence;
    use crate::diagnostics::Location;

    fn construct_char_sequence(input: &str) -> CharSequence {
        CharSequence::new(input)
    }

    fn new_loc(line: i64, column: i64) -> Location {
        Location { line, column }
    }

    #[test]
    fn test_empty_sequence() {
        let seq = construct_char_sequence("");
        assert_eq!(seq.look_at(0), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_single_char() {
        let mut seq = construct_char_sequence("a");
        assert_eq!(seq.look_at(0), Some('a'));
        assert_eq!(seq.look_at(0), Some('a'));
        seq.advance(1);
        assert_eq!(seq.location, new_loc(1, 2));
        assert_eq!(seq.look_at(0), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_multiple_chars() {
        let mut seq = construct_char_sequence("ab");
        assert_eq!(seq.look_at(0), Some('a'));
        assert_eq!(seq.look_at(1), Some('b'));
        seq.advance(2);
        assert_eq!(seq.location, new_loc(1, 3));
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_with_newline() {
        let mut seq = construct_char_sequence("abcdef\nhij");
        assert_eq!(seq.look_at(0), Some('a'));
        seq.advance(5);
        assert_eq!(seq.location, new_loc(1, 6));
        assert_eq!(seq.look_at(0), Some('f'));
        seq.advance(2);
        assert_eq!(seq.location, new_loc(2, 1));
        assert_eq!(seq.look_at(0), Some('h'));
    }
}
