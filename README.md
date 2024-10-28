# lox.rs

Rust implementation for the Lox language (from [Crafting Interpreters](https://craftinginterpreters.com/)).


## Execution

There are currently three possibilities:

o Interpreter Mode
o REPL Mode
o AST Mode


### Interpreter Mode

The *Interpreter* mode runs a complete script file and prints the output on `stdout`.
For example, assuming the follwing code in `script.lox`:

    var i = 1;
    print i;

Execute the code with:

    lox run script.lox


### REPL Mode

The *REPL* mode (read-eval-print loop) instantly executes individual statements entered on the command-line.
The REPL mode is started with:

    lox repl

This opens up a CLI prompt:

    lox> var i = 1;
    lox> fun some_func (p) { print p; }
    lox> some_func(i);
    1
    lox>


### AST Mode

The *AST* mode prints the Abstract Syntax Tree of an input program.
Assuming again the above `script.lox`, the AST gets printed on the command-line by:

    lox print-ast script.lox

The output is:

    Abstract Syntax Tree
    ====================
    (list
        (var-decl
            i
            (number 1)
        )
        (print
            (var i)
        )
    )
    ====================

This command additionally supports printing location information, i.e. the location where some entity (e.g. variable) was defined:

    lox print-ast --show-location script.lox

This then outputs the location information in brackets:

    Abstract Syntax Tree
    ====================
    (list [1:1-2:8]
        (var-decl [1:1-1:10]
            i
            (number 1) [1:9-1:9]
        )
        (print [2:1-2:8]
            (var i) [2:7-2:7]
        )
    )
    ====================