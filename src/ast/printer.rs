use crate::ast::{Expr, Literal};

fn print_expr (expr: &Expr) -> String {
    match expr {
        Expr::Literal(Literal::Boolean(value)) => {
            value.to_string()
        }
        _ => "<ELSE>".to_string()
    }
}

pub fn print_ast (expr: &Expr) -> String {
    print_expr(expr)
}