from graphviz import Digraph

def minimized_dfa_to_dot(model_output: str):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")  # Left to right

    # Initialize containers
    transitions = []
    start_state = None
    final_states = set()
    all_states = set()

    # 1. Parse input
    # Support both single and multiple string inputs (CSV line style)
    if isinstance(model_output, str):
        lines = [model_output]
    else:
        lines = model_output

    for line in lines:
        parts = [p.strip() for p in line.split(";") if p.strip()]
        state_transitions = []

        for part in parts:
            if part.startswith("in:"):
                start_state = part.replace("in:", "").strip()
            elif part.startswith("fi:"):
                final_states.update([s.strip() for s in part.replace("fi:", "").split(",") if s.strip()])
            else:
                # Transition format: A: a-->B, b-->C
                if ":" in part:
                    state_part, trans_part = part.split(":", 1)
                    state = state_part.strip()
                    all_states.add(state)
                    trans_items = [t.strip() for t in trans_part.split(",") if "-->" in t]
                    for item in trans_items:
                        label, target = item.split("-->")
                        label = label.strip()
                        target = target.strip()
                        transitions.append((state, target, label))
                        all_states.add(target)

    # 2. Start node (invisible) pointing to start state
    dot.node("start", shape="plaintext", label="")
    if start_state:
        dot.edge("start", start_state, label="start")

    # 3. Create nodes for all states
    for state in all_states:
        shape = "doublecircle" if state in final_states else "circle"
        dot.node(state, shape=shape)

    # 4. Create edges for transitions
    for src, dst, label in transitions:
        dot.edge(src, dst, label=label)

    return dot
