use crate::ast::{Expr, Literal};

fn get_indent_as_string(indent: i64) -> String {
    std::iter::repeat(" ")
        .take(indent as usize)
        .collect::<String>()
}

fn print_expr(expr: &Expr, indent: i64) -> String {
    const INDENT_NEXT_LEVEL: i64 = 2;

    match expr {
        Expr::Binary { lhs, op, rhs, loc } => {
            let lhs_str = print_expr(lhs, indent + INDENT_NEXT_LEVEL);
            let op_str = op.to_string();
            let rhs_string = print_expr(rhs, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}{} [{}]\n{}\n{}",
                get_indent_as_string(indent),
                op_str,
                loc.to_string(),
                lhs_str,
                rhs_string
            )
        }
        Expr::Unary { op, expr, loc } => {
            let op_str = op.to_string();
            let expr_str = print_expr(expr, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}{} [{}]\n{}",
                get_indent_as_string(indent),
                op_str,
                loc.to_string(),
                expr_str
            )
        }
        Expr::Grouping { expr, loc } => {
            let expr_str = print_expr(expr, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}() [{}]\n{}",
                get_indent_as_string(indent),
                loc.to_string(),
                expr_str
            )
        }
        Expr::Literal {
            literal: Literal::Boolean(value),
            loc,
        } => format!(
            "{}{} [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
        Expr::Literal {
            literal: Literal::Nil,
            loc,
        } => format!("{}nil [{}]", get_indent_as_string(indent), loc.to_string()),
        Expr::Literal {
            literal: Literal::String(value),
            loc,
        } => format!(
            "{}\"{}\" [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
        Expr::Literal {
            literal: Literal::Number(value),
            loc,
        } => format!(
            "{}{} [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
    }
}

pub fn print_ast(expr: &Expr) -> String {
    print_expr(expr, 0)
}
