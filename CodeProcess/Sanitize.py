import re

def remove_comments(code: str):
    pattern = r"""
                                ##  --------- COMMENT ---------
               //.*?$           ##  Start of // .... comment
             |                  ##
               /\*              ##  Start of /* ... */ comment
               [^*]*\*+         ##  Non-* followed by 1-or-more *'s
               (                ##
                 [^/*][^*]*\*+  ##
               )*               ##  0-or-more things which don't start with / but do end with '*'
               /                ##  End of /* ... */ comment
             |                  ##  -OR-  various things which aren't comments:
               (                ##
                                ##  ------ " ... " STRING ------
                 "              ##  Start of " ... " string
                 (              ##
                   \\.          ##  Escaped char
                 |              ##  -OR-
                   [^"\\]       ##  Non "\ characters
                 )*             ##
                 "              ##  End of " ... " string
               |                ##  -OR-
                                ##
                                ##  ------ ' ... ' STRING ------
                 '              ##  Start of ' ... ' string
                 (              ##
                   \\.          ##  Escaped char
                 |              ##  -OR-
                   [^'\\]       ##  Non '\ characters
                 )*             ##
                 '              ##  End of ' ... ' string
               |                ##  -OR-
                                ##
                                ##  ------ ANYTHING ELSE -------
                 .              ##  Anything other char
                 [^/"'\\]*      ##  Chars which doesn't start a comment, string or escape
               )                ##
        """
    regex = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(code) if m.group(2)]
    return "".join(noncomments)
