#!/usr/bin/env python3
"""
Streamlit front-end for the Tomasulo simulator.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd
import streamlit as st

from tomasulo_core import (
    Instruction,
    TomasuloSimulator,
    SAMPLE_PROGRAM_TEXT,
    build_sample_program,
    default_memory,
    default_registers,
    parse_program,
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "register_defaults" not in st.session_state:
        st.session_state["register_defaults"] = default_registers()
    if "memory_defaults" not in st.session_state:
        st.session_state["memory_defaults"] = default_memory()
    if "program_text" not in st.session_state:
        st.session_state["program_text"] = SAMPLE_PROGRAM_TEXT.strip()
    if "simulator" not in st.session_state:
        st.session_state["simulator"] = TomasuloSimulator(
            program=build_sample_program(),
            register_init=st.session_state["register_defaults"],
            memory_init=st.session_state["memory_defaults"],
        )


def replace_simulator(program_text: str) -> None:
    st.session_state["program_text"] = program_text
    st.session_state["simulator"] = TomasuloSimulator(
        program=parse_program(program_text),
        register_init=st.session_state["register_defaults"],
        memory_init=st.session_state["memory_defaults"],
    )


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

def render_header(sim: TomasuloSimulator) -> None:
    st.title("Tomasulo Algorithm Simulator")
    st.caption("Interactive dynamic scheduling demo with reservation stations and CDB.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cycle", sim.cycle)
    col2.metric("Issued", sum(1 for instr in sim.instructions if instr.issue_cycle is not None))
    col3.metric("Finished", "Yes" if sim.is_finished() else "No")


def render_instruction_editor() -> None:
    st.subheader("Instruction Input")
    st.caption("ISA syntax: `LD F6, 34(R2)` | `MULTD F0, F2, F4` | etc.")

    editor_col, buttons_col = st.columns([4, 1])
    with editor_col:
        text = st.text_area(
            "Program",
            value=st.session_state["program_text"],
            height=140,
            label_visibility="collapsed",
        )
    with buttons_col:
        st.markdown("**Program Actions**")
        if st.button("Apply", use_container_width=True, type="primary"):
            try:
                parse_program(text)
            except ValueError as exc:
                st.error(f"Failed to parse instructions: {exc}")
            else:
                replace_simulator(text.strip())
                st.rerun()
        if st.button("Load Example", use_container_width=True):
            replace_simulator(SAMPLE_PROGRAM_TEXT.strip())
            st.rerun()


def render_controls(sim: TomasuloSimulator) -> None:
    st.subheader("Execution Controls")
    col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1])
    if col1.button("Step", use_container_width=True):
        sim.step()
        st.rerun()
    step_count = col2.number_input("Run cycles", min_value=1, max_value=200, value=10, step=1, label_visibility="collapsed")
    col2.caption("Cycles / burst")
    if col3.button(f"Run ×{int(step_count)}", use_container_width=True):
        for _ in range(step_count):
            if sim.is_finished():
                break
            sim.step()
        st.rerun()
    if col4.button("Reset", type="secondary", use_container_width=True):
        sim.reset()
        st.rerun()


def render_tables(sim: TomasuloSimulator) -> None:
    instructions_df = pd.DataFrame([instruction_row(idx, instr) for idx, instr in enumerate(sim.instructions)])
    rs_df = pd.DataFrame([reservation_row(rs) for rs in sim.reservation_stations])
    reg_df = pd.DataFrame(register_rows(sim.register_file, sim.register_status))
    mem_df = pd.DataFrame(memory_rows(sim.memory))

    st.subheader("Machine State")
    top_left, top_right = st.columns((3, 2))
    with top_left:
        st.markdown("#### Instructions")
        st.dataframe(instructions_df, use_container_width=True, hide_index=True, height=260)
    with top_right:
        st.markdown("#### Reservation Stations")
        st.dataframe(rs_df, use_container_width=True, hide_index=True, height=260)

    bottom_left, bottom_right = st.columns((2, 2))
    with bottom_left:
        st.markdown("#### Registers")
        st.dataframe(reg_df, use_container_width=True, hide_index=True, height=240)
    with bottom_right:
        st.markdown("#### Memory")
        st.dataframe(mem_df, use_container_width=True, hide_index=True, height=240)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def instruction_row(idx: int, instr: Instruction) -> Dict[str, str]:
    return {
        "#": idx + 1,
        "Op": instr.op,
        "Dest": instr.dest or "",
        "Src1": instr.src1 or "",
        "Src2": instr.src2 or "",
        "Mem": instr.mem_str,
        "Issue": instr.issue_cycle if instr.issue_cycle is not None else "",
        "Exec Start": instr.exec_start if instr.exec_start is not None else "",
        "Exec End": instr.exec_end if instr.exec_end is not None else "",
        "Write": instr.write_cycle if instr.write_cycle is not None else "",
    }


def reservation_row(rs) -> Dict[str, str]:
    return {
        "Name": rs.name,
        "Busy": "Yes" if rs.busy else "No",
        "Op": rs.op or "",
        "Vj": f"{rs.Vj:.3f}" if rs.Vj is not None else "",
        "Vk": f"{rs.Vk:.3f}" if rs.Vk is not None else "",
        "Qj": rs.Qj or "",
        "Qk": rs.Qk or "",
        "Addr": rs.address if rs.address is not None else "",
        "Remain": rs.remaining if rs.started else "",
    }


def register_rows(registers: Dict[str, float], status: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"Name": name, "Value": f"{value:.3f}", "Qi": status.get(name) or ""}
        for name, value in sorted(registers.items())
    ]


def memory_rows(memory: Dict[int, float]) -> List[Dict[str, str]]:
    if not memory:
        return []
    addresses = sorted(memory.keys())
    blocks: List[Dict[str, str]] = []
    for addr in addresses:
        blocks.append({"Address": addr, "Value": f"{memory[addr]:.3f}"})
    return blocks


def main() -> None:
    st.set_page_config(page_title="Tomasulo Simulator", layout="wide")
    init_state()
    simulator: TomasuloSimulator = st.session_state["simulator"]
    render_header(simulator)
    with st.container():
        render_instruction_editor()
    with st.container():
        render_controls(simulator)
    with st.container():
        render_tables(simulator)


if __name__ == "__main__":
    main()


