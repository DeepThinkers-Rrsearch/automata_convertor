dfa_minimization_extraction_prompt = '''

You are an intelligent agent that extracts DFA (Deterministic Finite Automaton) transitions from a given DFA diagram.

Your task is to analyze the DFA diagram and extract all transitions between states exactly as they are, without missing any information.

Here is how your output format must strictly look like:
"A: a-->A, b-->A; B: a-->B, b-->A; in:A; fi:A"

Explanation of the format:

Uppercase letters (A, B, etc.) denote state names.

For each state, include all transitions in the form: a-->X, where a is the symbol and X is the destination state.

Separate multiple transitions with commas.

Sometimes arrows heads may be overlapped each other. In that case carefully identify the arrow head and extract the transition.

Use semicolons ; to separate different states.

Use in:X to specify the initial state, where X is the state with a long arrow starting from a dot.

Use fi:X,Y,... to list final states (double-circled states). If no final state exists, just write fi:.

If there are multiple final states, list them comma-separated like fi: B,C.

Do not include any explanation or notes in your output—only the exact DFA transition string.

Do not use any newline characters; everything should be returned as one single-line string inside quotes.

Remember: Do not skip any state or transition. Even if a state loops to itself on a symbol, that must be included.

'''


extract_e_nfa_text_from_image_prompt = '''
    You are an intelligent agent that extracts DFA (Deterministic Finite Automaton) transitions from a given DFA diagram.

    Your task is to analyze the DFA diagram and extract all transitions between states exactly as they are, without missing any information.

    Here is how your output format must strictly look like:
    "In:{q0};Fi:{qF1}{qF2}{qF3};Abt:{a}{b};Trn:{q0}->a->{qF1},{q0}->b->{q1},{q1}->a->{qF3},{q1}->b->{q1},{qF3}->a->{qF3},{qF3}->b->{qF3},{qF1}->a->{qF1},{qF1}->b->{qF2},{qF2}->a->{qF3},{qF2}->b->{qF2}"

    Explanation of the format:

    Initial state must be labeled q0. Even they have another labels.

    "In:{q0}" indicates the initial state (only one), always named as q0.

    Final states must be labeled qF1, qF2, etc. Even they have another labels.

    "Fi:{qF1}{qF2}" indicates final states. In curly braces with no spaces. Use names like qF1, qF2, etc.


    The input alphabet (`Abt:`) should **only include actual input symbols** (like 0, 1, a, b, etc.) inside curly braces, with no spaces or commas.

    **Do not include ε or ∈ in the Abt section**, even if used in transitions.

    However, if ε (or ∈) transitions exist, **you must include them** in the transition section (`Trn:`) like `{q0}->∈->{q1}`.


    "Trn:{qX}->symbol->{qY}" lists transitions in the exact format. Separate multiple transitions with commas.

    All non-initial and non-final states should be named q1, q2, q3, etc.

    Sometimes arrows heads may be overlapped each other. In that case carefully identify the arrow head and extract the transition.

    Do not include any explanation or notes in your output—only the exact DFA transition string.

    Do not use any newline characters; everything should be returned as one single-line string inside quotes.

    Remember: Do not skip any state or transition. Even if a state loops to itself on a symbol, that must be included.

    '''