from graphviz import Digraph

def pda_output_to_dot(transitions: list[str]):
    """
    Draw the PDA with Graphviz and write a PNG image.

    Parameters
    ----------
    transitions : list[str]
        Lines like "delta(q0, a, Z) -> (q0, PUSH)".
    filename : str | None
        If given, used as the base name *without extension*.
        Default = first call decides a name from q0→qf path.
    out_dir : str
        Folder where images are written. Created on demand.
    """
    import os, re

    #os.makedirs(out_dir, exist_ok=True)

    # ---------- build the graph ---------------------------------------------
    dot = Digraph(format="png")
    dot.attr(rankdir="LR", fontsize="12")

    # draw final state in double-circle
    dot.node("qf", shape="doublecircle")
    states_seen: set[str] = {"qf"}

    pattern = r"delta\((\w+),\s*([\wε\&]),\s*(\w)\)\s*->\s*\((\w+),\s*(\w+)\)"

    for line in transitions:
        m = re.match(pattern, line)
        if not m:
            continue
        frm, inp, stk, to, act = m.groups()

        # add nodes lazily
        if frm not in states_seen:
            dot.node(frm, shape="circle")
            states_seen.add(frm)
        if to not in states_seen:
            shape = "doublecircle" if to == "qf" else "circle"
            dot.node(to, shape=shape)
            states_seen.add(to)

        edge_label = f"{inp}, {stk} / {act}"
        dot.edge(frm, to, label=edge_label)

    return dot

    # ---------- write file ---------------------------------------------------
    # if filename is None:
    #     filename = "pda_" + "_".join(list(states_seen)[:3])  # fallback

    # out_path = os.path.join(out_dir, filename)
    # dot.render(out_path, cleanup=True)   # writes out_path + ".png"
    # print(f"PDA diagram saved → {out_path}.png")