use crate::diagnostics::LocationSpan;

pub mod printer;

pub enum UnaryOperator {
    Minus,
    Not,
}

impl ToString for UnaryOperator {
    fn to_string(&self) -> String {
        match *self {
            UnaryOperator::Minus => String::from("-"),
            UnaryOperator::Not => String::from("!"),
        }
    }
}

pub enum BinaryOperator {
    Add,
    Substract,
    Multiply,
    Divide,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl ToString for BinaryOperator {
    fn to_string(&self) -> String {
        match *self {
            BinaryOperator::Add => String::from("+"),
            BinaryOperator::Substract => String::from("-"),
            BinaryOperator::Multiply => String::from("*"),
            BinaryOperator::Divide => String::from("/"),
            BinaryOperator::Equal => String::from("=="),
            BinaryOperator::NotEqual => String::from("!="),
            BinaryOperator::GreaterThan => String::from(">"),
            BinaryOperator::GreaterThanOrEqual => String::from(">="),
            BinaryOperator::LessThan => String::from("<"),
            BinaryOperator::LessThanOrEqual => String::from("<="),
        }
    }
}

pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

pub enum Expr {
    Binary {
        lhs: Box<Expr>,
        op: BinaryOperator,
        rhs: Box<Expr>,
        loc: LocationSpan,
    },
    Unary {
        op: UnaryOperator,
        expr: Box<Expr>,
        loc: LocationSpan,
    },
    Literal {
        literal: Literal,
        loc: LocationSpan,
    },
    Grouping {
        expr: Box<Expr>,
        loc: LocationSpan,
    }
}

impl Expr {
    pub fn get_loc (&self) -> &LocationSpan {
        match self {
            Expr::Binary { loc, .. } => loc,
            Expr::Unary { loc, .. } => loc,
            Expr::Grouping { loc, .. } => loc,
            Expr::Literal { loc, .. } => loc,
        }
    }
}
