from graphviz import Digraph
import re

def dfa_output_to_dot(model_output: str):
    # Parse the sections
    parts = {}
    model_output = model_output.strip()
    if model_output.startswith('"') and model_output.endswith('"'):
        model_output = model_output[1:-1]

    for section in model_output.split(';'):
        if not section:
            continue
        key, value = section.split(':', 1)
        parts[key.strip()] = value.strip()

    # Initial state
    initial_states = re.findall(r'\{(.*?)\}', parts.get('In', ''))
    initial_state = initial_states[0] if initial_states else None

    # Final states
    final_states = re.findall(r'\{(.*?)\}', parts.get('Fi', ''))

    # Alphabet (not used in graph, but can validate transitions)
    alphabet = re.findall(r'\{(.*?)\}', parts.get('Abt', ''))

    # Transitions
    transitions_str = parts.get('Trn', '')
    transition_list = [t.strip() for t in transitions_str.split(',') if t.strip()]
    transitions = []
    all_states = set()
    for t in transition_list:
        match = re.match(r'\{(.*?)\}->(.*?)\->\{(.*?)\}', t)
        if match:
            src, label, dst = match.groups()
            transitions.append((src, dst, label))
            all_states.add(src)
            all_states.add(dst)

    # Build the Graphviz DFA
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")
    dot.node("start", shape="plaintext", label="")

    # Initial state edge
    if initial_state:
        dot.edge("start", initial_state, label="start")

    # State nodes (draw all, mark final later)
    for st in all_states:
        dot.node(st, shape="circle")
    # Mark final states
    for fs in final_states:
        dot.node(fs, shape="doublecircle")

    # Add transitions
    for src, dst, label in transitions:
        dot.edge(src, dst, label=label)

    return dot
