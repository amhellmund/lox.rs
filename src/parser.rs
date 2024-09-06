use crate::{ast::{Expr, Literal}, scanner::Token};

pub fn parse (_tokens: &Vec<Token>) -> Expr {
    Expr::Literal(
        Literal::Boolean(true)
    )
}