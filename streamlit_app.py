#!/usr/bin/env python3
"""
Streamlit front-end for the Tomasulo Simulator.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# Core Tomasulo Simulator

@dataclass
class Instruction:
    op: str
    dest: Optional[str] = None
    src1: Optional[str] = None
    src2: Optional[str] = None
    offset: Optional[int] = None
    base: Optional[str] = None
    issue_cycle: Optional[int] = None
    exec_start: Optional[int] = None
    exec_end: Optional[int] = None
    write_cycle: Optional[int] = None

    def clone(self) -> "Instruction":
        return copy.deepcopy(self)

    @property
    def mem_str(self) -> str:
        if self.offset is not None and self.base:
            return f"{self.offset}({self.base})"
        return ""

@dataclass
class ReservationStation:
    name: str
    rs_type: str
    busy: bool = False
    op: Optional[str] = None
    Vj: Optional[float] = None
    Vk: Optional[float] = None
    Qj: Optional[str] = None
    Qk: Optional[str] = None
    A: Optional[int] = None
    address: Optional[int] = None
    dest: Optional[str] = None
    instruction_idx: Optional[int] = None
    remaining: int = 0
    started: bool = False
    result: Optional[float] = None

    def reset(self) -> None:
        self.busy = False
        self.op = None
        self.Vj = None
        self.Vk = None
        self.Qj = None
        self.Qk = None
        self.A = None
        self.address = None
        self.dest = None
        self.instruction_idx = None
        self.remaining = 0
        self.started = False
        self.result = None

@dataclass
class CycleEvent:
    """Records what happened during a cycle."""
    cycle: int
    issued: Optional[Tuple[int, str, str]] = None  # (instr_idx, instr_text, rs_name)
    issue_stall: Optional[Tuple[int, str, str]] = None  # (instr_idx, instr_text, reason)
    exec_started: List[Tuple[str, str]] = None  # [(rs_name, instr_text)]
    exec_completed: List[Tuple[str, str]] = None  # [(rs_name, instr_text)]
    wrote_result: Optional[Tuple[str, str, str]] = None  # (rs_name, register, value)
    
    def __post_init__(self):
        if self.exec_started is None:
            self.exec_started = []
        if self.exec_completed is None:
            self.exec_completed = []

class TomasuloSimulator:
    RS_TEMPLATES: Dict[str, List[str]] = {
        "Load": ["Load1", "Load2", "Load3"],
        "Store": ["Store1", "Store2"],
        "Add": ["Add1", "Add2", "Add3"],
        "Mult": ["Mult1", "Mult2"],
    }

    DEFAULT_LATENCIES: Dict[str, int] = {
        "Load": 2,
        "Store": 2,
        "Add": 2,
        "Sub": 2,
        "Mult": 10,
        "Div": 20,
    }

    def __init__(
        self,
        program: List[Instruction],
        register_init: Optional[Dict[str, float]] = None,
        memory_init: Optional[Dict[int, float]] = None,
        latencies: Optional[Dict[str, int]] = None,
    ) -> None:
        self.original_program = [instr.clone() for instr in program]
        self.register_init = register_init or {}
        self.memory_init = memory_init or {}
        self.latencies = latencies or self.DEFAULT_LATENCIES.copy()
        self.reservation_stations: List[ReservationStation] = []
        self.instructions: List[Instruction] = []
        self.register_file: Dict[str, float] = {}
        self.register_status: Dict[str, Optional[str]] = {}
        self.memory: Dict[int, float] = {}
        self.cycle: int = 0
        self.issue_ptr: int = 0
        self.events: List[CycleEvent] = []
        self.current_event: Optional[CycleEvent] = None
        self.reset()

    def reset(self) -> None:
        self.cycle = 0
        self.issue_ptr = 0
        self.instructions = [instr.clone() for instr in self.original_program]
        self.register_file = copy.deepcopy(self.register_init)
        self.memory = copy.deepcopy(self.memory_init)
        self.register_status = {reg: None for reg in self.register_file}
        self.reservation_stations = [
            ReservationStation(name, rs_type)
            for rs_type, names in self.RS_TEMPLATES.items()
            for name in names
        ]
        self.events = []
        self.current_event = None

        for instr in self.instructions:
            for reg in (instr.dest, instr.src1, instr.src2, instr.base):
                if reg and reg not in self.register_file:
                    self.register_file[reg] = 0.0
                    self.register_status[reg] = None

    def step(self) -> bool:
        if self.is_finished():
            return False
        self.cycle += 1
        self.current_event = CycleEvent(cycle=self.cycle)
        self._write_results()
        self._advance_executions()
        self._issue_instruction()
        self.events.append(self.current_event)
        return True

    def is_finished(self) -> bool:
        return all(instr.write_cycle is not None for instr in self.instructions)

    def _format_instruction(self, instr: Instruction) -> str:
        """Format instruction for display in events."""
        if instr.dest and instr.src1 and instr.src2:
            return f"{instr.op} {instr.dest}, {instr.src1}, {instr.src2}"
        elif instr.dest and instr.mem_str:
            return f"{instr.op} {instr.dest}, {instr.mem_str}"
        elif instr.src1 and instr.mem_str:
            return f"{instr.op} {instr.src1}, {instr.mem_str}"
        return instr.op

    def _write_results(self) -> None:
        ready = [
            rs
            for rs in self.reservation_stations
            if rs.busy and rs.rs_type != "Store" and rs.result is not None
        ]
        if not ready:
            return
        ready.sort(
            key=lambda rs: (
                self.instructions[rs.instruction_idx or 0].issue_cycle or 0,
                rs.name,
            )
        )
        rs = ready[0]
        result = rs.result
        instr = self.instructions[rs.instruction_idx or 0]
        instr.write_cycle = self.cycle
        
        # Record event
        if self.current_event and rs.dest:
            self.current_event.wrote_result = (rs.name, rs.dest, f"{result:.3f}")
        
        if rs.dest and self.register_status.get(rs.dest) == rs.name:
            self.register_file[rs.dest] = result
            self.register_status[rs.dest] = None
        for other in self.reservation_stations:
            if other.Qj == rs.name:
                other.Vj = result
                other.Qj = None
            if other.Qk == rs.name:
                other.Vk = result
                other.Qk = None
        rs.reset()

    def _advance_executions(self) -> None:
        for rs in self.reservation_stations:
            if not rs.busy:
                continue
            instr = self.instructions[rs.instruction_idx or 0]
            if not rs.started:
                if not self._operands_ready(rs):
                    continue
                rs.started = True
                rs.remaining = self._latency_for(rs)
                if rs.rs_type in {"Load", "Store"}:
                    base_val = rs.Vj or 0.0
                    rs.address = int(base_val + (rs.A or 0))
                instr.exec_start = instr.exec_start or self.cycle
                
                # Record event
                if self.current_event:
                    self.current_event.exec_started.append(
                        (rs.name, self._format_instruction(instr))
                    )
                    
            if rs.remaining > 0:
                rs.remaining -= 1
                if rs.remaining == 0:
                    instr.exec_end = self.cycle
                    
                    # Record event
                    if self.current_event:
                        self.current_event.exec_completed.append(
                            (rs.name, self._format_instruction(instr))
                        )
                    
                    if rs.rs_type == "Store":
                        self._store_to_memory(rs, instr)
                    else:
                        rs.result = self._compute_result(rs)

    def _issue_instruction(self) -> None:
        if self.issue_ptr >= len(self.instructions):
            return
        instr = self.instructions[self.issue_ptr]
        rs_type = self._op_to_type(instr.op)
        station = self._find_free_station(rs_type)
        
        if station is None:
            # Record structural hazard
            if self.current_event:
                reason = f"No free {rs_type} reservation station"
                self.current_event.issue_stall = (
                    self.issue_ptr,
                    self._format_instruction(instr),
                    reason
                )
            return
            
        station.busy = True
        station.op = instr.op
        station.dest = instr.dest
        station.instruction_idx = self.issue_ptr
        station.started = False
        if instr.dest:
            self.register_status.setdefault(instr.dest, None)
            self.register_status[instr.dest] = station.name
        if instr.base is not None:
            self._capture_operand(station, instr.base, position="j")
            station.A = instr.offset or 0
        if instr.src1:
            target = "k" if station.rs_type == "Store" else "j"
            self._capture_operand(station, instr.src1, position=target)
        if instr.src2:
            self._capture_operand(station, instr.src2, position="k")
        instr.issue_cycle = self.cycle
        
        # Record event
        if self.current_event:
            self.current_event.issued = (
                self.issue_ptr,
                self._format_instruction(instr),
                station.name
            )
        
        self.issue_ptr += 1

    def _capture_operand(
        self, station: ReservationStation, register: str, position: str = "j"
    ) -> None:
        self.register_file.setdefault(register, 0.0)
        self.register_status.setdefault(register, None)
        qi = self.register_status.get(register)
        if qi:
            if position == "j":
                station.Qj = qi
            else:
                station.Qk = qi
        else:
            if position == "j":
                station.Vj = self.register_file.get(register, 0.0)
            else:
                station.Vk = self.register_file.get(register, 0.0)

    def _operands_ready(self, station: ReservationStation) -> bool:
        if station.rs_type == "Load":
            return station.Qj is None
        if station.rs_type == "Store":
            return station.Qj is None and station.Qk is None
        return station.Qj is None and station.Qk is None

    def _latency_for(self, station: ReservationStation) -> int:
        op = (station.op or "").upper()
        if op.startswith("LD"):
            return self.latencies.get("Load", 2)
        elif op.startswith(("SD", "ST")):
            return self.latencies.get("Store", 2)
        elif op.startswith("ADD"):
            return self.latencies.get("Add", 2)
        elif op.startswith("SUB"):
            return self.latencies.get("Sub", 2)
        elif op.startswith("MUL"):
            return self.latencies.get("Mult", 10)
        elif op.startswith("DIV"):
            return self.latencies.get("Div", 20)
        else:
            # Default for unknown operations
            return 2

    def _compute_result(self, station: ReservationStation) -> float:
        op = (station.op or "").upper()
        if op.startswith("ADD"):
            return (station.Vj or 0.0) + (station.Vk or 0.0)
        if op.startswith("SUB"):
            return (station.Vj or 0.0) - (station.Vk or 0.0)
        if op.startswith("MUL"):
            return (station.Vj or 0.0) * (station.Vk or 0.0)
        if op.startswith("DIV"):
            divisor = station.Vk or 1.0
            return (station.Vj or 0.0) / divisor if divisor != 0 else float("inf")
        if op.startswith("LD"):
            return self.memory.get(station.address or 0, 0.0)
        raise ValueError(f"Unsupported operation {station.op}")

    def _store_to_memory(self, station: ReservationStation, instr: Instruction) -> None:
        value = station.Vk or 0.0
        addr = station.address or 0
        self.memory[addr] = value
        instr.write_cycle = self.cycle
        station.reset()

    def _find_free_station(self, rs_type: str) -> Optional[ReservationStation]:
        for station in self.reservation_stations:
            if station.rs_type == rs_type and not station.busy:
                return station
        return None

    def _op_to_type(self, op: str) -> str:
        op = op.upper()
        if op.startswith("LD"):
            return "Load"
        if op.startswith(("SD", "ST")):
            return "Store"
        if op.startswith(("AD", "SU")):
            return "Add"
        return "Mult"

SAMPLE_PROGRAM_TEXT = """\
LD F6, 34(R2)
LD F2, 45(R3)
MULTD F0, F2, F4
SUBD F8, F6, F2
DIVD F10, F0, F6
ADDD F6, F8, F2
"""

MEM_ADDR_RE = re.compile(r"^\s*(-?\d+)\((\w+)\)\s*$")
ALLOWED_PREFIXES = ("LD", "ST", "SD", "AD", "SU", "MU", "DI")

def parse_program(text: str) -> List[Instruction]:
    instructions: List[Instruction] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = re.split(r"[,\s]+", line)
        if not tokens:
            continue
        op = tokens[0].upper()
        if not op.startswith(ALLOWED_PREFIXES):
            raise ValueError(f"Line {line_no}: unsupported opcode '{op}'")
        try:
            if op.startswith("LD"):
                dest, mem = tokens[1], tokens[2]
                offset, base = _parse_mem_operand(mem)
                instructions.append(Instruction(op=op, dest=dest, base=base, offset=offset))
            elif op.startswith(("SD", "ST")):
                src, mem = tokens[1], tokens[2]
                offset, base = _parse_mem_operand(mem)
                instructions.append(Instruction(op=op, src1=src, base=base, offset=offset))
            else:
                dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                instructions.append(Instruction(op=op, dest=dest, src1=src1, src2=src2))
        except IndexError as exc:
            raise ValueError(f"Line {line_no}: missing operand in '{raw_line}'") from exc
    return instructions

def _parse_mem_operand(token: str) -> Tuple[int, str]:
    match = MEM_ADDR_RE.match(token)
    if not match:
        raise ValueError(f"Invalid memory operand '{token}' (expected offset(base))")
    offset, base = match.groups()
    return int(offset), base

def build_sample_program() -> List[Instruction]:
    return parse_program(SAMPLE_PROGRAM_TEXT)

def default_registers() -> Dict[str, float]:
    return {
        "F0": 1.0,
        "F2": 2.0,
        "F4": 4.0,
        "F6": 6.0,
        "F8": 8.0,
        "F10": 10.0,
        "R1": 0.0,
        "R2": 0.0,
        "R3": 0.0,
    }

def default_memory() -> Dict[int, float]:
    return {
        34: 3.0,
        45: 5.0,
    }

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
    if "latencies" not in st.session_state:
        st.session_state["latencies"] = TomasuloSimulator.DEFAULT_LATENCIES.copy()
    if "simulator" not in st.session_state:
        st.session_state["simulator"] = TomasuloSimulator(
            program=build_sample_program(),
            register_init=st.session_state["register_defaults"],
            memory_init=st.session_state["memory_defaults"],
            latencies=st.session_state["latencies"],
        )
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "auto_speed" not in st.session_state:
        st.session_state["auto_speed"] = 1.0


def replace_simulator(program_text: str) -> None:
    st.session_state["program_text"] = program_text
    st.session_state["simulator"] = TomasuloSimulator(
        program=parse_program(program_text),
        register_init=st.session_state["register_defaults"],
        memory_init=st.session_state["memory_defaults"],
        latencies=st.session_state["latencies"],
    )

# UI rendering
def render_header(sim: TomasuloSimulator) -> None:
    st.title("Tomasulo Algorithm Simulator")
    st.caption("Interactive dynamic scheduling demo with reservation stations and CDB.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cycle", sim.cycle)
    col2.metric("Issued", sum(1 for instr in sim.instructions if instr.issue_cycle is not None))
    col3.metric("Finished", "Yes" if sim.is_finished() else "No")

def render_current_status(sim: TomasuloSimulator) -> None:
    """Display current cycle status and any structural hazards."""
    if sim.cycle == 0:
        return
        
    st.subheader("Current Cycle Status")
    
    # Show next instruction to issue
    if sim.issue_ptr < len(sim.instructions):
        next_instr = sim.instructions[sim.issue_ptr]
        rs_type = sim._op_to_type(next_instr.op)
        free_station = sim._find_free_station(rs_type)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Next to issue:** Instruction #{sim.issue_ptr + 1}: {sim._format_instruction(next_instr)}")
        with col2:
            if free_station is None:
                st.error(f"🚫 **STRUCTURAL HAZARD**: No free {rs_type} reservation station available")
            else:
                st.success(f"✓ {rs_type} station available ({free_station.name})")
    
    # Show last cycle events
    if sim.events:
        last_event = sim.events[-1]
        
        event_lines = []
        if last_event.issued:
            idx, instr, rs = last_event.issued
            event_lines.append(f"✓ Issued: Instruction #{idx + 1} → {rs}")
        if last_event.issue_stall:
            idx, instr, reason = last_event.issue_stall
            event_lines.append(f"⚠️ Stalled: Instruction #{idx + 1} - {reason}")
        if last_event.exec_started:
            for rs, instr in last_event.exec_started:
                event_lines.append(f"▶️ Started execution: {rs} ({instr})")
        if last_event.exec_completed:
            for rs, instr in last_event.exec_completed:
                event_lines.append(f"⏹️ Completed execution: {rs}")
        if last_event.wrote_result:
            rs, reg, val = last_event.wrote_result
            event_lines.append(f"📤 Wrote result: {rs} → {reg} = {val}")
        
        if event_lines:
            st.markdown("**Last Cycle Events:**")
            for line in event_lines:
                st.markdown(f"- {line}")

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

def render_latency_config() -> None:
    """Render latency configuration panel."""
    with st.expander("⚙️ Latency Configuration", expanded=False):
        st.caption("Configure execution latencies (in cycles) for each operation type")
        
        col1, col2, col3 = st.columns(3)
        
        latencies_changed = False
        new_latencies = st.session_state["latencies"].copy()
        
        with col1:
            st.markdown("**Memory Operations**")
            load_lat = st.number_input(
                "Load (LD)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Load"],
                step=1,
                key="load_latency"
            )
            if load_lat != st.session_state["latencies"]["Load"]:
                new_latencies["Load"] = load_lat
                latencies_changed = True
                
            store_lat = st.number_input(
                "Store (SD/ST)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Store"],
                step=1,
                key="store_latency"
            )
            if store_lat != st.session_state["latencies"]["Store"]:
                new_latencies["Store"] = store_lat
                latencies_changed = True
        
        with col2:
            st.markdown("**Arithmetic Operations**")
            add_lat = st.number_input(
                "Add (ADDD)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Add"],
                step=1,
                key="add_latency"
            )
            if add_lat != st.session_state["latencies"]["Add"]:
                new_latencies["Add"] = add_lat
                latencies_changed = True
                
            sub_lat = st.number_input(
                "Subtract (SUBD)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Sub"],
                step=1,
                key="sub_latency"
            )
            if sub_lat != st.session_state["latencies"]["Sub"]:
                new_latencies["Sub"] = sub_lat
                latencies_changed = True
        
        with col3:
            st.markdown("**Complex Operations**")
            mult_lat = st.number_input(
                "Multiply (MULTD)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Mult"],
                step=1,
                key="mult_latency"
            )
            if mult_lat != st.session_state["latencies"]["Mult"]:
                new_latencies["Mult"] = mult_lat
                latencies_changed = True
                
            div_lat = st.number_input(
                "Divide (DIVD)",
                min_value=1,
                max_value=50,
                value=st.session_state["latencies"]["Div"],
                step=1,
                key="div_latency"
            )
            if div_lat != st.session_state["latencies"]["Div"]:
                new_latencies["Div"] = div_lat
                latencies_changed = True
        
        # Apply button and reset to defaults
        button_col1, button_col2, _ = st.columns([1, 1, 2])
        
        with button_col1:
            if st.button("Apply Latencies", use_container_width=True, type="primary"):
                st.session_state["latencies"] = new_latencies
                # Recreate simulator with new latencies
                st.session_state["simulator"] = TomasuloSimulator(
                    program=st.session_state["simulator"].original_program,
                    register_init=st.session_state["register_defaults"],
                    memory_init=st.session_state["memory_defaults"],
                    latencies=new_latencies,
                )
                st.session_state["simulator"].reset()
                st.success("Latencies updated and simulator reset!")
                st.rerun()
        
        with button_col2:
            if st.button("Reset to Defaults", use_container_width=True):
                st.session_state["latencies"] = TomasuloSimulator.DEFAULT_LATENCIES.copy()
                st.session_state["simulator"] = TomasuloSimulator(
                    program=st.session_state["simulator"].original_program,
                    register_init=st.session_state["register_defaults"],
                    memory_init=st.session_state["memory_defaults"],
                    latencies=st.session_state["latencies"],
                )
                st.session_state["simulator"].reset()
                st.rerun()

def render_controls(sim: TomasuloSimulator) -> None:
    st.subheader("Execution Controls")
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1.2, 1, 1])
    
    # Start/Stop button
    if st.session_state["is_running"]:
        if col1.button("⏸ Stop", use_container_width=True, type="primary"):
            st.session_state["is_running"] = False
            st.rerun()
    else:
        if col1.button("▶️ Start", use_container_width=True, type="primary"):
            if not sim.is_finished():
                st.session_state["is_running"] = True
                st.rerun()
    
    # Step button
    if col2.button("Step", use_container_width=True, disabled=st.session_state["is_running"]):
        sim.step()
        st.rerun()
    
    # Multi-step controls
    step_count = col3.number_input("Run cycles", min_value=1, max_value=200, value=10, step=1, label_visibility="collapsed", disabled=st.session_state["is_running"])
    col3.caption("Cycles / burst")
    if col4.button(f"Run ×{int(step_count)}", use_container_width=True, disabled=st.session_state["is_running"]):
        for _ in range(step_count):
            if sim.is_finished():
                break
            sim.step()
        st.rerun()
    
    # Reset button
    if col5.button("Reset", type="secondary", use_container_width=True, disabled=st.session_state["is_running"]):
        sim.reset()
        st.session_state["is_running"] = False
        st.rerun()
    
    # Speed control
    if st.session_state["is_running"]:
        speed = st.slider(
            "Simulation Speed",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state["auto_speed"],
            step=0.1,
            format="%.1fx",
            help="Control how fast the simulation runs"
        )
        st.session_state["auto_speed"] = speed

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

def render_gantt_chart(sim: TomasuloSimulator) -> None:
    st.subheader("Execution Timeline")
    
    data = []
    for i, instr in enumerate(sim.instructions):
        label = f"#{i+1} {instr.op}"
        
        # 1. Issue Phase (From Issue to Exec Start)
        if instr.issue_cycle and instr.exec_start:
            data.append({"Instruction": label, "Stage": "Issue (Wait)", "Start": instr.issue_cycle, "End": instr.exec_start})
            
        # 2. Execution Phase (From Exec Start to Exec End)
        if instr.exec_start and instr.exec_end:
            data.append({"Instruction": label, "Stage": "Execute", "Start": instr.exec_start, "End": instr.exec_end})
            
        # 3. Write Result Phase (From Exec End to Write)
        if instr.exec_end and instr.write_cycle:
             # If write happens same cycle as exec ends, add small buffer to make it visible
            end_time = instr.write_cycle if instr.write_cycle > instr.exec_end else instr.exec_end + 0.1
            data.append({"Instruction": label, "Stage": "Write (CDB)", "Start": instr.exec_end, "End": end_time})

    if not data:
        st.info("Run the simulation to see the timeline.")
        return

    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Start', title='Cycle'),
        x2='End',
        y=alt.Y('Instruction', sort=None), # Keep instruction order
        color=alt.Color('Stage', scale=alt.Scale(domain=['Issue (Wait)', 'Execute', 'Write (CDB)'], range=['#f0ad4e', '#5bc0de', '#5cb85c'])),
        tooltip=['Instruction', 'Stage', 'Start', 'End']
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)

# Serialization helpers
def instruction_row(idx: int, instr: Instruction) -> Dict[str, str]:
    return {
        "#": idx + 1,
        "Op": instr.op,
        "Dest": instr.dest or "",
        "Src1": instr.src1 or "",
        "Src2": instr.src2 or "",
        "Mem": instr.mem_str,
        "Issue":str(instr.issue_cycle) if instr.issue_cycle is not None else "",
        "Exec Start": str(instr.exec_start) if instr.exec_start is not None else "",
        "Exec End": str(instr.exec_end) if instr.exec_end is not None else "",
        "Write": str(instr.write_cycle) if instr.write_cycle is not None else "",
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
        "Addr": str(rs.address) if rs.address is not None else "",
        "Remain": str(rs.remaining) if rs.started else "",
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
        render_latency_config()
    with st.container():
        render_controls(simulator)
    with st.container():
        render_current_status(simulator)
    with st.container():
        render_tables(simulator)
    with st.container():
        render_gantt_chart(simulator)
    
    if st.session_state["is_running"] and not simulator.is_finished():
        delay_ms = int(1000 / st.session_state["auto_speed"])
        simulator.step()
        time.sleep(delay_ms / 1000.0)
        st.rerun()
    elif st.session_state["is_running"] and simulator.is_finished():
        st.session_state["is_running"] = False
        st.rerun()

if __name__ == "__main__":
    main()
