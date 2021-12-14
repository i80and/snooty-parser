from typing import List

from hypothesis import given, settings
from hypothesis.strategies import from_regex, one_of, sampled_from, text

from .minirst import BlockParser, TokenType, tokenize
from .util_test import ast_to_testing_string, check_ast_testing_string


def pretty_print(xml_text: str) -> str:
    import xml.dom.minidom

    dom = xml.dom.minidom.parseString(xml_text)
    return dom.toprettyxml()  # type: ignore


def test_lexer() -> None:
    should_lex = lambda name: name != "code-block"
    token_types = [
        t.type.name
        for t in tokenize(
            """======
insert
======

Definition
----------

.. contents-word:: On this page
   :option: 1
 \x20
   .. foobar::

      some body
      some more

      * foobar

        baz

      * .. foobar::

           content

        | :ref:`db.collection.updateOne() <updateOne-example-agg>`
        | * noNewScope

   some more body

- * entry 1
  * entry 2

\x20""",
            should_lex,
        )
    ]
    tokens_by_line: List[List[str]] = []
    for token in token_types:
        if token is TokenType.LineStart.name:
            tokens_by_line.append([])
        tokens_by_line[-1].append(token)

    assert token_types.count(TokenType.Indent.name) == token_types.count(
        TokenType.Dedent.name
    )

    ideal_lines = [
        ["LineStart", "Rule"],
        ["LineStart", "Text"],
        ["LineStart", "Rule"],
        ["LineStart"],
        ["LineStart", "Text"],
        ["LineStart", "Rule"],
        ["LineStart"],
        [
            "LineStart",
            "DotDot",
            "Text",
            "Dash",
            "Text",
            "DoubleColon",
            "TextWhitespace",
            "Text",
            "TextWhitespace",
            "Text",
            "TextWhitespace",
            "Text",
        ],
        ["LineStart", "Indent", "FieldName", "TextWhitespace", "Text"],
        ["LineStart"],
        ["LineStart", "DotDot", "Text", "DoubleColon"],
        ["LineStart"],
        ["LineStart", "Indent", "Text", "TextWhitespace", "Text"],
        ["LineStart", "Text", "TextWhitespace", "Text"],
        ["LineStart"],
        ["LineStart", "Asterisks", "TextWhitespace", "Indent", "Text"],
        ["LineStart"],
        ["LineStart", "Text"],
        ["LineStart"],
        [
            "LineStart",
            "Dedent",
            "Asterisks",
            "TextWhitespace",
            "Indent",
            "DotDot",
            "Text",
            "DoubleColon",
        ],
        ["LineStart"],
        ["LineStart", "Indent", "Text"],
        ["LineStart"],
        [
            "LineStart",
            "Dedent",
            "Pipe",
            "TextWhitespace",
            "Role",
            "Ticks",
            "Text",
            "Text",
            "Text",
            "Text",
            "Text",
            "Text",
            "Text",
            "TextWhitespace",
            "Text",
            "Text",
            "Dash",
            "Text",
            "Dash",
            "Text",
            "Text",
            "Rule",
        ],
        ["LineStart", "Pipe", "TextWhitespace", "Asterisks", "TextWhitespace", "Text"],
        ["LineStart"],
        [
            "LineStart",
            "Dedent",
            "Dedent",
            "Text",
            "TextWhitespace",
            "Text",
            "TextWhitespace",
            "Text",
        ],
        ["LineStart"],
        [
            "LineStart",
            "Dedent",
            "Dash",
            "TextWhitespace",
            "Indent",
            "Asterisks",
            "TextWhitespace",
            "Indent",
            "Text",
            "TextWhitespace",
            "Text",
        ],
        [
            "LineStart",
            "Dedent",
            "Asterisks",
            "TextWhitespace",
            "Indent",
            "Text",
            "TextWhitespace",
            "Text",
        ],
        ["LineStart"],
        ["LineStart", "Dedent", "Dedent"],
    ]

    for actual, ideal in zip(tokens_by_line, ideal_lines):
        assert actual == ideal, f"{actual} = {ideal}"

    list(
        tokenize(
            """
   .. code-block:: javascript

      var mapFunction2 = function() {
         for (var idx = 0; idx < this.items.length; idx++) {
            var key = this.items[idx].sku;
            var value = { count: 1, qty: this.items[idx].qty };

            emit(key, value);
         }
     };
    """,
            should_lex,
        )
    )

    list(
        tokenize(
            """
.. authrole::

   - word

 \xa0 .. warning::
""",
            should_lex,
        )
    )

    list(
        tokenize(
            """
.. code-block:: cpp

   OID make_an_id() {
     OID x = OID::gen();
     return x;
   }
""",
            should_lex,
        )
    )

    list(
        tokenize(
            """
    .. authrole:: backup

    .. todo: word
                - system.new_users
                - system.backup_users
                - system.version
                Do we want to document these?
    """,
            should_lex,
        )
    )

    list(
        tokenize(
            """
- Words

  .. warning::

     .. See :issue:`SERVER-9562` for more information.

.. code-block:: javascript

   {
     <field1>: <value1>
   }
""",
            should_lex,
        )
    )


@given(
    one_of(
        sampled_from([".. ", "foo"]), from_regex(":[^\\S:]{1,5}:`"), text("[]*:`._-\n")
    )
)
@settings(max_examples=1000)
def test_lexer_hypothesis(s: str) -> None:
    list(tokenize(s, lambda name: name != "code-block"))


def test_parse_headings_and_directives() -> None:
    test_string = """
======
insert
======

.. default-domain:: mongodb

.. contents:: On this page
   :local:
   :backlinks: none
   :depth: 1
   :class: singlecol

Command Definition
------------------

.. dbcommand:: insert

   The :dbcommand:`insert` command inserts one or more documents and
   returns a document containing the status of all inserts. The insert
   methods provided by the MongoDB drivers use this command internally.

   .. note::

      This is a nested directive.

   The command has the following syntax:
"""
    parser = BlockParser()
    print(pretty_print(ast_to_testing_string(parser.ingest_text(test_string))))
    check_ast_testing_string(
        parser.ingest_text(test_string),
        """
<root fileid=".">
    <section>
        <heading id="insert"><text>insert</text></heading>
        <directive name="default-domain"><text>mongodb</text></directive>
        <directive name="contents" local="" backlinks="none" depth="1" class="singlecol"><text> On this page</text></directive>
        <section>
            <heading id="command-definition"><text>Command Definition</text></heading>
            <directive name="dbcommand"><text> insert</text>
                <paragraph><text>The :dbcommand:`insert` command inserts one or more documents and returns a document containing the status of all inserts. The insert methods provided by the MongoDB drivers use this command internally.</text></paragraph>
                <directive name="note"><text></text>
                    <paragraph><text>This is a nested directive.</text></paragraph>
                </directive>
                <paragraph><text>The command has the following syntax:</text></paragraph>
            </directive>
        </section>
    </section>
</root>
    """,
    )


def test_unordered_lists() -> None:
    test_string = """
.. list-table::

   * - Field

     - Description
       some text

       some more text

   * - .. note:: foobar

     - .. note:: foobar

          * List
"""
    parser = BlockParser()
    print(pretty_print(ast_to_testing_string(parser.ingest_text(test_string))))


# def main() -> None:
#     root = Path("/home/heli/work/docs/source/")
#     paths = list(root.glob("**/*.rst")) + list(root.glob("**/*.txt"))
#     # paths = [root / "core" / "document.txt"]
#     print(len(paths))
#     for path in paths:
#         # print()
#         print(path.as_posix())
#         # list(tokenize(path.read_text(), lambda name: name != "code-block"))
#     # print([t.type.name for t in tokenize(TEST_FILE, lambda name: name != "code-block")])
