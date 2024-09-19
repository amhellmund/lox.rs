# lox.rs

Rust implementation for the Lox language (from [Crafting Interpreters](https://craftinginterpreters.com/)).


## Execution

There are currently two possibilities to run the Lox interpreter: either in file or REPL mode.

### File Mode

Given a Lox file, the interpreter gets started in file mode by:

    lox <file>

To work interactively with Lox, start the interpreter in REPL mode by:

    lox


## Example

An iteractive Lox session in REPL mode could look like

    lox> 1 + 2
    Result: 3
    lox> "a" + "b"
    Result: "ab"
    lox> 1 + "2"
    Error: Binary operator '+' only supported for number operands: given number and string [repl@1:1-1:5]