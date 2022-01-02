from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Match,
    MutableSequence,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from . import n, util

# Goals:
# * Exact character-range node & diagnostic source tracking
# * Straight-line faster than docutils for our needs
# * Simpler logic to be quickly ported into other languages
# * Exclude features we don't want
# Bonus:
# * Support some nested markup


# Supported block elements:
# * Headings, both underline & overline variants
# * Footnotes
# * Substitutions
# * Horizontal Lines
# * Directives
# * Lists, either * or -
# * Field lists
# * Line block

PAT_RULE_TEXT = r"(?P<rule>(?:(?:=+)|(?:-+)|(?:~+)|(?:`+)|(?:\^+)|(?:'+)|(?:!+))$)"
PAT_FOOTNOTE_TEXT = r"(?P<footnote>\[(?:(?:\d+)|\*|\#(?:[\w_]*))\])"
PAT_DOTDOT_TEXT = r"(?P<dotdot>\.\.\x20)"
PAT_DOUBLECOLON_TEXT = r"(?P<double_colon>::)"
PAT_ROLE_NAME_TEXT = r"(?::(?P<role_name>[\w_]+):(?=`))"
PAT_TICKS_TEXT = r"(?P<ticks>`{1,2})"
PAT_FIELD_NAME_TEXT = r"(?::(?P<field_name>[\w_-]+):)"
PAT_ASTERISKS_TEXT = r"(?P<asterisks>\*)"
PAT_DASH_TEXT = r"(?P<dash>\-)"
PAT_LINE_START = r"(?P<line_start>^\x20*\n?)"
PAT_PIPE = r"(?P<pipe>\|)"
PAT_TEXT_WHITESPACE = r"(?P<text_whitespace>\x20+)"
PAT_TEXT_TEXT = r"(?P<text>[\w]+|.)"

PAT_TEXT = "|".join(
    [
        PAT_LINE_START,
        PAT_RULE_TEXT,
        PAT_TEXT_WHITESPACE,
        PAT_FOOTNOTE_TEXT,
        PAT_DOTDOT_TEXT,
        PAT_DOUBLECOLON_TEXT,
        PAT_ROLE_NAME_TEXT,
        PAT_TICKS_TEXT,
        PAT_FIELD_NAME_TEXT,
        PAT_ASTERISKS_TEXT,
        PAT_DASH_TEXT,
        PAT_PIPE,
        PAT_TEXT_TEXT,
    ]
)
# print(PAT_TEXT)
PAT_TOKENS = re.compile(
    PAT_TEXT,
    re.MULTILINE | re.VERBOSE,
)


class Token(NamedTuple):
    type: TokenType
    match: Match[str]


class TokenType(enum.Enum):
    Rule = enum.auto()
    Footnote = enum.auto()
    DotDot = enum.auto()
    DoubleColon = enum.auto()
    Role = enum.auto()
    Ticks = enum.auto()
    FieldName = enum.auto()
    Asterisks = enum.auto()
    Dash = enum.auto()
    LineStart = enum.auto()
    Pipe = enum.auto()
    Text = enum.auto()
    TextWhitespace = enum.auto()

    # Synthetic, generated during lexing
    Indent = enum.auto()
    Dedent = enum.auto()


TOKEN_TYPES_TO_CODES = {
    TokenType.Rule: "=",
    TokenType.Footnote: "#",
    TokenType.DotDot: ".",
    TokenType.DoubleColon: ":",
    TokenType.Role: "R",
    TokenType.Ticks: "`",
    TokenType.FieldName: "F",
    TokenType.Asterisks: "*",
    TokenType.Dash: "-",
    TokenType.LineStart: "L",
    TokenType.Pipe: "|",
    TokenType.Text: "T",
    TokenType.TextWhitespace: " ",
    TokenType.Indent: "I",
    TokenType.Dedent: "D",
}


PAT_TO_TOKEN: Dict[str, TokenType] = {
    "rule": TokenType.Rule,
    "footnote": TokenType.Footnote,
    "dotdot": TokenType.DotDot,
    "double_colon": TokenType.DoubleColon,
    "role_name": TokenType.Role,
    "ticks": TokenType.Ticks,
    "field_name": TokenType.FieldName,
    "asterisks": TokenType.Asterisks,
    "dash": TokenType.Dash,
    "line_start": TokenType.LineStart,
    "pipe": TokenType.Pipe,
    "text_whitespace": TokenType.TextWhitespace,
    "text": TokenType.Text,
}


def tokenize(text: str, should_lex_directive: Callable[[str], bool]) -> Iterator[Token]:
    text = text.rstrip(" ")
    text = text.replace("\xa0", " ")

    indent_stack: List[int] = [0]
    line_so_far: List[Token] = []
    wait_until_indent_exit = False
    watching_for_double_colon = False

    def line_is_directive() -> Optional[str]:
        """If the line so far seems to name a directive, return its name."""
        name: List[str] = []
        for token in reversed(line_so_far):
            if token.type in {TokenType.Text, TokenType.Dash}:
                name.append(token.match.group(0))
            else:
                break

        if name:
            return "".join(name[::-1])

        return None

    for match in PAT_TOKENS.finditer(text):
        assert match.lastgroup is not None, "tokenizing regex issue"
        token_type = PAT_TO_TOKEN[match.lastgroup]

        token = Token(token_type, match)
        yield token

        all_text = match.group(0)
        if token_type is TokenType.LineStart:
            watching_for_double_colon = False

        if token_type is TokenType.LineStart and "\n" not in all_text:
            line_so_far.clear()

            new_indent = len(all_text)
            current_indent = indent_stack[-1]

            if wait_until_indent_exit and new_indent == current_indent:
                wait_until_indent_exit = False
            elif new_indent < current_indent:
                try:
                    found_level = indent_stack.index(new_indent)
                except ValueError:
                    print(repr(all_text), indent_stack)
                    # XXX We need to raise a diagnostic and gracefully recover
                    raise

                level_delta = len(indent_stack) - found_level
                del indent_stack[found_level + 1 :]
                for _ in range(level_delta - 1):
                    yield Token(TokenType.Dedent, match)
            elif not wait_until_indent_exit:
                if new_indent > current_indent:
                    indent_stack.append(new_indent)
                    yield Token(TokenType.Indent, match)
        elif watching_for_double_colon and token_type is TokenType.DoubleColon:
            # Don't lex this indentation scope if the directive name demands it not be lexed.
            # This is a giant honking layer violation. Dunno what to tell you. Needs to be done.
            directive_name = line_is_directive()
            if directive_name is not None and should_lex_directive(directive_name):
                wait_until_indent_exit = False
                watching_for_double_colon = False
        elif not wait_until_indent_exit:
            if token_type is TokenType.TextWhitespace:
                # These list tokens can introduce new indentation scopes
                if all(
                    (
                        (
                            (t.type in {TokenType.Asterisks, TokenType.Dash})
                            or t.type in {TokenType.LineStart, TokenType.TextWhitespace}
                        )
                        for t in line_so_far
                    )
                ):
                    new_length = sum(len(t.match.group(0)) for t in line_so_far) + len(
                        all_text
                    )
                    indent_stack.append(new_length)
                    yield Token(TokenType.Indent, match)
            elif token_type is TokenType.DotDot:
                wait_until_indent_exit = True
                watching_for_double_colon = True

        line_so_far.append(token)

    while len(indent_stack) > 1:
        yield Token(TokenType.Dedent, match)
        indent_stack.pop()

    assert indent_stack == [0]


@dataclass
class TokenList:
    __slots__ = ("tokens", "current")

    tokens: List[Token]
    current: int

    def __getitem__(self, i: int) -> Optional[Token]:
        try:
            return self.tokens[i]
        except IndexError:
            return None

    def take_until(
        self, start: int, predicate: Callable[[Token], bool]
    ) -> Optional[Tuple[int, Sequence[Token]]]:
        i = start

        while i < len(self.tokens):
            if predicate(self.tokens[i]):
                return i, self.tokens[start:i]
            i += 1

        return None


@dataclass
class DirectiveParser:
    __slots__ = (
        "block_parser",
        "tokens",
        "token_codes",
        "match",
        "options",
        "argument_slice",
        "body_slice",
    )

    block_parser: BlockParser
    tokens: Sequence[Token]
    token_codes: str

    match: Match[str]
    options: Dict[str, slice]
    argument_slice: Optional[slice]
    body_slice: Optional[slice]

    def argument_text(self) -> Optional[str]:
        if self.argument_slice is None:
            return None
        return "".join(t.match.group(0) for t in self.tokens[self.argument_slice])

    @property
    def have_argument(self) -> bool:
        return self.argument_slice is not None

    def body_text(self) -> Optional[str]:
        if self.body_slice is None:
            return None
        return "".join(t.match.group(0) for t in self.tokens[self.body_slice])

    def parse_body(self) -> MutableSequence[n.Node]:
        if self.body_slice is not None:
            child = self.block_parser.create_child()
            return child.ingest(
                self.tokens,
                self.token_codes,
                self.body_slice.start,
                self.body_slice.stop,
            )
        return []

    def option_text(self, option: str) -> Optional[str]:
        try:
            option_slice = self.options[option]
        except KeyError:
            return None
        return "".join(t.match.group(0) for t in self.tokens[option_slice])

    @property
    def have_body(self) -> bool:
        return self.body_slice is not None


@dataclass
class DirectiveDefinition:
    parse: DirectiveHandler
    lex: bool


DirectiveHandler = Callable[
    [DirectiveDefinition, Tuple[str, str], DirectiveParser], List[n.Node]
]


@dataclass
class Domain:
    directives: Dict[str, DirectiveDefinition] = field(default_factory=dict)


def parse_linenos(term: str, max_val: int) -> List[Tuple[int, int]]:
    """Parse a comma-delimited list of line numbers and ranges."""
    results: List[Tuple[int, int]] = []
    if not term.strip():
        return []
    for term in term.strip().split(","):
        parts = term.split("-", 1)
        lower = int(parts[0])
        higher = int(parts[1]) if len(parts) == 2 else lower
        if lower < 0 or higher < 0:
            raise ValueError(
                f"Invalid line number specification: {term}. Expects non-negative integers."
            )
        elif lower > max_val or higher > max_val:
            raise ValueError(
                f"Invalid line number specification: {term}. Expects maximum value of {max_val}."
            )
        elif lower > higher:
            raise ValueError(
                f"Invalid line number specification: {term}. Expects {lower} < {higher}."
            )

        results.append((lower, higher))

    return results


class InlineParser:
    def __init__(self) -> None:
        self.role_definitions: Dict[Tuple[str, str], object] = {}

    def ingest(self, tokens: Sequence[Token], start: int, end: int) -> None:
        pass

class BlockParser:
    # Some of what we need to do is really expressible as regular grammars. We can
    # exploit this by representing each token as a single character code 💪🏻
    PAT_HEADER_OVER_AND_UNDER = re.compile(
        r"(?:L|D)(?P<line1>=)L(?P<title>[^L]+)L(?P<line2>=)"
    )
    PAT_HEADER_UNDER = re.compile(r"(?:L|D)(?P<title>[^L]+)L(?P<line1>=)")
    PAT_LINE_BREAKS = re.compile(r"LL+")
    PAT_DIRECTIVE = re.compile(
        r"(?:[LID\*\-])*\.(?P<name>[T-]+): *(?P<argument>[^L]*)(?P<body_start>L+I)?"
    )
    PAT_FIELDLIST = re.compile(r"(?:(?:I|L)(?P<name>F) *(?P<contents>[^L]*))+?")
    PAT_LIST = re.compile(r"L*(?:[DI]?(?P<ch>[\*-])) ?")

    def __init__(
        self,
        default_domain: Optional[str] = None,
        domain_resolution_sequence: Sequence[str] = ("mongodb", "std", ""),
    ) -> None:
        self.heading_levels: List[str] = []

        def handle_default(
            definition: DirectiveDefinition,
            name: Tuple[str, str],
            parser: DirectiveParser,
        ) -> List[n.Node]:
            argument_text = parser.argument_text()
            argument_nodes = (
                [] if not argument_text else [n.Text((-1,), argument_text)]
            )
            children = parser.parse_body()
            options: Dict[str, str] = {
                k: parser.option_text(k) or "" for k, v in parser.options.items()
            }
            return [n.Directive((-1,), children, name[0], name[1], argument_nodes, options)]  # type: ignore

        def handle_code_block(
            definition: DirectiveDefinition,
            name: Tuple[str, str],
            parser: DirectiveParser,
        ) -> List[n.Node]:
            body_text = parser.body_text() or ""
            lang = parser.argument_text()
            if lang is not None:
                lang = lang.strip()
            raw_lineno_start = parser.option_text("lineno_start")
            lineno_start = (
                int(raw_lineno_start) if raw_lineno_start is not None else None
            )
            raw_emphasize_lines = parser.option_text("emphasize_lines")
            emphasize_lines = (
                parse_linenos(raw_emphasize_lines, len(body_text))
                if raw_emphasize_lines is not None
                else None
            )
            return [
                n.Code(
                    (-1,),
                    lang,
                    parser.option_text("caption"),
                    "copyable" in parser.options,
                    emphasize_lines,
                    body_text,
                    "linenos" in parser.options,
                    lineno_start,
                )
            ]

        self.domains: Dict[str, Domain] = {
            "std": Domain(
                {
                    "default-domain": DirectiveDefinition(handle_default, True),
                    "contents": DirectiveDefinition(handle_default, True),
                    "code-block": DirectiveDefinition(handle_code_block, False),
                    "list-table": DirectiveDefinition(handle_default, True),
                    "include": DirectiveDefinition(handle_default, True),
                    "note": DirectiveDefinition(handle_default, True),
                }
            ),
            "mongodb": Domain(
                {
                    "dbcommand": DirectiveDefinition(handle_default, True),
                    "data": DirectiveDefinition(handle_default, True),
                }
            ),
        }

        name_sequence: Sequence[str] = domain_resolution_sequence
        if default_domain is not None:
            name_sequence = (default_domain, *name_sequence)
        self.domain_sequence = [
            self.domains[domain_name]
            for domain_name in name_sequence
            if domain_name in self.domains
        ]

    def create_child(self) -> BlockParser:
        child = BlockParser()
        child.domains = self.domains
        child.heading_levels = self.heading_levels
        return child

    def ingest_inline(
        self, tokens: Sequence[Token], token_codes: str, start: int, end: int
    ) -> MutableSequence[n.Node]:
        result: List[str] = []
        skipping_whitespace = False
        i = start
        just_saw_whitespace = False
        while i < end:
            t = tokens[i]
            if t.type is TokenType.TextWhitespace or t.type is TokenType.LineStart:
                text = " "
            else:
                text = t.match.group(0)

            result.append(text)
            i += 1

        value = "".join(result).strip()
        if value:
            return [n.Text((-1,), value)]
        return []

    def ingest(
        self, tokens: Sequence[Token], token_codes: str, start: int, end: int
    ) -> MutableSequence[n.Node]:
        stack: List[n.Parent[n.Node]] = [n.Parent((0,), [])]
        list_context: List[str] = []
        current = start

        def find_list_items(list_start: int, list_character: TokenType) -> Iterator[slice]:
            i = list_start
            indent_level = 1
            while indent_level != 0 and i < end - 1:
                i += 1

                if tokens[i].type is TokenType.Indent:
                    indent_level += 1
                elif tokens[i].type is TokenType.Dedent:
                    indent_level -= 1
                elif indent_level == 1 and tokens[i].type is list_character:
                    yield slice(list_start, i)
                    list_start = i

            yield slice(list_start, i)
            yield slice(i, i)


        def chomp_indentation_scope(start: int) -> int:
            """Return the token index where the indentation scope introduced by start ends."""
            i = start
            indent_level = 1
            while indent_level != 0 and i < end - 1:
                i += 1
                if tokens[i].type is TokenType.Indent:
                    indent_level += 1
                elif tokens[i].type is TokenType.Dedent:
                    indent_level -= 1

            return i

        def pop_paragraph() -> None:
            if isinstance(stack[-1], n.Paragraph):
                top = stack.pop()
                stack[-1].children.append(top)

        def parse_field_list(start_from: int) -> Tuple[Dict[str, slice], int]:
            options: Dict[str, slice] = {}
            while True:
                field_match = self.PAT_FIELDLIST.match(token_codes, start_from)
                if not field_match:
                    return options, start_from
                field_name = tokens[field_match.start("name")].match.group("field_name")
                options[field_name] = slice(
                    field_match.start("contents"), field_match.end("contents")
                )
                start_from = field_match.end()

            return options, start_from

        def try_consume_header() -> bool:
            nonlocal current

            for pattern in (self.PAT_HEADER_OVER_AND_UNDER, self.PAT_HEADER_UNDER):
                match = pattern.match(token_codes, current - 1, end)
                if match is not None:
                    pop_paragraph()

                    children = self.ingest_inline(
                        tokens, token_codes, match.start("title"), match.end("title")
                    )
                    heading_id = util.make_html5_id(
                        "".join(c.get_text() for c in children).strip()
                    ).lower()
                    groupdict = match.groupdict()
                    char_type_1 = groupdict["line1"]
                    char_type_2 = groupdict.get("line2")
                    if char_type_2 is not None and char_type_1 != char_type_2:
                        raise Exception("Heading type mismatch")
                    if char_type_1 in self.heading_levels:
                        raise NotImplementedError("")
                    node = n.Heading((-1,), children, heading_id)  # type: ignore
                    section = n.Section((-1,), [node])  # type: ignore
                    stack.append(section)
                    current = match.end(0)
                    return True

            return False

        def try_consume_break() -> bool:
            nonlocal current

            match = self.PAT_LINE_BREAKS.match(token_codes, current, end)
            if match is not None:
                pop_paragraph()
                current = match.end(0)
                return True

            return False

        def try_consume_list() -> bool:
            nonlocal current

            match = self.PAT_LIST.match(token_codes, current - 1, end)
            if not match:
                return False

            pop_paragraph()

            list_kind = tokens[match.start("ch")].type
            list_node = n.ListNode((-1,), [], n.ListEnumType.unordered, None)  # type: ignore
            for list_slice in find_list_items(match.end("ch"), list_kind):
                current = list_slice.stop
                if list_slice.start is list_slice.stop:
                    break
                child_parser = self.create_child()
                children = child_parser.ingest(
                    tokens,
                    token_codes,
                    list_slice.start + 2,
                    list_slice.stop,
                )
                list_node.children.append(n.ListNodeItem((-1,), children))  # type: ignore

            stack[-1].children.append(list_node)
            return True

        def try_consume_directive() -> bool:
            nonlocal current

            match = self.PAT_DIRECTIVE.match(token_codes, current, end)
            if match is None:
                return False

            body_start_group = match.group("body_start")
            body_start_end = match.end("body_start")
            i = body_start_end if body_start_group is not None else match.end()

            field_list_end = i
            options: Dict[str, slice] = {}

            if body_start_group is not None:
                options, field_list_end = parse_field_list(i - 1)
                i = chomp_indentation_scope(field_list_end)

            name = util.split_domain(
                "".join(
                    t.match.group(0)
                    for t in tokens[match.start("name") : match.end("name")]
                )
            )

            directive_definition = self.lookup_directive(name[0], name[1])
            if directive_definition is None:
                raise Exception("Couldn't find definition for " + str(name))
            else:
                pop_paragraph()
                stack[-1].children.extend(
                    directive_definition.parse(
                        directive_definition,
                        name,
                        DirectiveParser(
                            self,
                            tokens,
                            token_codes,
                            match,
                            options,
                            slice(match.start("argument"), match.end("argument")),
                            slice(field_list_end, i) if body_start_group else None,
                        ),
                    )
                )

            current = i + 1
            return True

        def try_consume_paragraph() -> bool:
            nonlocal current

            i = current
            while i < end:
                if tokens[i].type is TokenType.LineStart:
                    if i + 1 < end and tokens[i + 1].type is TokenType.LineStart:
                        break
                i += 1

            if i > current:
                pop_paragraph()

                children = self.ingest_inline(tokens, token_codes, current, i)
                if children:
                    node = n.Paragraph((tokens[current].match.start(),), children)  # type: ignore
                    stack.append(node)
                current = i
                return True

            return False

        while current < end:
            if try_consume_header():
                continue
            if try_consume_break():
                continue
            if try_consume_list():
                continue
            if try_consume_directive():
                continue
            if try_consume_paragraph():
                continue

            assert (
                False
            ), f"No pattern match at {current} for {repr([t.type.name for t in tokens[current:end]])}"

        while len(stack) > 1:
            top = stack.pop()
            stack[-1].children.append(top)

        return stack[0].children

    def ingest_text(self, text: str) -> n.Root:
        tokens = list(tokenize(text, self._should_lex_directive))
        token_codes = "".join([TOKEN_TYPES_TO_CODES[t.type] for t in tokens])
        children = self.ingest(tokens, token_codes, 0, len(tokens))
        root = n.Root((0,), children, n.FileId(""), {})  # type: ignore
        return root

    def lookup_directive(
        self, domain_name: str, directive_name: str
    ) -> Optional[DirectiveDefinition]:
        if domain_name:
            return self.domains[domain_name].directives.get(directive_name, None)

        for domain in self.domain_sequence:
            if directive_name in domain.directives:
                return domain.directives.get(directive_name, None)

        return self.domains["std"].directives.get("note")

        return None

    def _should_lex_directive(self, name: str) -> bool:
        domain_name, directive_name = util.split_domain(name)
        result = self.lookup_directive(domain_name, directive_name)
        if result:
            return result.lex

        return False


def main() -> None:
    from pathlib import Path

    root = Path("/home/heli/work/docs/source/")
    paths = list(root.glob("**/*.rst")) + list(root.glob("**/*.txt"))
    # paths = [root / "core" / "document.txt"]
    print(len(paths))
    for path in paths:
        # print()
        # print(path.as_posix())
        # list(tokenize(path.read_text(), lambda name: name != "code-block"))
        BlockParser().ingest_text(path.read_text())
    # print([t.type.name for t in tokenize(TEST_FILE, lambda name: name != "code-block")])


if __name__ == "__main__":
    main()
