from graphviz import Digraph

def epsilon_nfa_to_dot(model_output: str):

    # 1. Split into comma-separated pieces
    segments = [seg.strip() for seg in model_output.split(",") if seg.strip()]

    transitions = []      # will hold tuples (src, dst, label)
    start_state = None    # e.g. "q0"
    final_states = set()  # e.g. {"q3", ...}

    for seg in segments:
        has_start = "[start]" in seg
        has_end   = "[end]" in seg

        # Remove any “[start]” / “[end]” tokens before splitting on “--”
        clean_seg = seg.replace("[start]", "").replace("[end]", "").strip()

        # Splitting on “--” yields: [maybe_state_or_tag, label, next_state, label, next_state, …]
        parts = [p.strip() for p in clean_seg.split("--") if p.strip()]
        # e.g. ["q0", "ε", "q1", "0", "q2", "ε", "q3"]

        # Extract transitions by walking through parts in steps of 2:
        #   (parts[0] --parts[1]--> parts[2]),
        #   (parts[2] --parts[3]--> parts[4]), etc.
        for i in range(0, len(parts) - 2, 2):
            src   = parts[i]
            label = parts[i + 1]
            dst   = parts[i + 2]
            transitions.append((src, dst, label))

        # If “[start]” appeared in this segment, the first part is the start state.
        if has_start:
            first_state = parts[0]
            start_state = first_state

        # If “[end]” appeared in this segment, the last part is a final state.
        if has_end:
            last_state = parts[-1]
            final_states.add(last_state)

    # 2. Build the Graphviz Digraph
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")             # left-to-right layout
    dot.node("start", shape="plaintext", label="")  # invisible “start” node

    # If we found a start_state, draw an edge from the invisible start node to it
    if start_state:
        dot.edge("start", start_state, label="start")

    # Ensure we register every state (so isolated states still appear as circles)
    all_states = set()
    for (src, dst, _) in transitions:
        all_states.add(src)
        all_states.add(dst)
    for st in all_states:
        # By default, every state is drawn as a circle. If it's final, override later.
        dot.node(st, shape="circle")

    # 3. Add each transition arrow
    for (src, dst, label) in transitions:
        # If dst is in final_states, we’ll convert it to doublecircle below.
        dot.edge(src, dst, label=label)

    # 4. Mark final states with doublecircle
    for fs in final_states:
        # Redraw node with shape=doublecircle (Graphviz will override the previous node shape)
        dot.node(fs, shape="doublecircle")

    # 5. Render to file (e.g. “epsilon_nfa_diagram.png”)
    return dot
