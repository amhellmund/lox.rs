use crate::ast::{Expr, Literal};

fn print_expr (expr: &Expr) -> String {
    match expr {
        Expr::Binary {lhs, op, rhs} => {
            let lhs_str = print_expr(lhs);
            let op_str = op.to_string();
            let rhs_string = print_expr(rhs);
            format!("<{} {} {}>", op_str, lhs_str, rhs_string)
        },
        Expr::Unary {op, expr} => {
            let op_str = op.to_string();
            let expr_str = print_expr(expr);
            format!("<{} {}>", op_str, expr_str)
        },
        Expr::Grouping { expr } => {
            let expr_str = print_expr(expr);
            format!("({})", expr_str)
        },
        Expr::Literal(Literal::Boolean(value)) => {
            value.to_string()
        },
        Expr::Literal(Literal::Nil) => {
            "nil".to_string()
        }
        Expr::Literal(Literal::String(value)) => {
            String::from(value)
        },
        Expr::Literal(Literal::Number(value)) => {
            value.to_string()
        }
    }
}

pub fn print_ast (expr: &Expr) -> String {
    print_expr(expr)
}