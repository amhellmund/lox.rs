pub mod printer;

pub enum UnaryOperator {
    Minus,
    Not,
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
    },
    Unary {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    Literal(Literal),
}