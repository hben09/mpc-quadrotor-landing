#!/usr/bin/env python3
"""Read a simulation trace CSV and plot delta x and delta y vs time.

Usage:
  python plot.py --data-folder data1
  python plot.py --input data/data1/simulation_trace_*.csv --show
"""
from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLOTS_DIR = SCRIPT_DIR / 'plot'
NUMERIC_COLUMNS = ('time_s', 'drone_x', 'drone_y', 'ground_x', 'ground_y')
TEXT_COLUMNS = ('phase',)
OPTIONAL_NUMERIC_COLUMNS = ('armed',)
PLOT_PARAMETER_NAMES = ('mpc_horizon', 'mpc_q_diag', 'mpc_qf_diag')


def collect_phase_segments(times: list[float], phases: list[str]) -> list[tuple[float, float, str]]:
	if len(times) != len(phases):
		raise ValueError('time and phase series must have the same length')
	if not times:
		return []

	segments: list[tuple[float, float, str]] = []
	start_idx = 0
	for idx in range(1, len(phases)):
		if phases[idx] != phases[start_idx]:
			segments.append((times[start_idx], times[idx], phases[start_idx]))
			start_idx = idx
	segments.append((times[start_idx], times[-1], phases[start_idx]))
	return segments


def collect_arm_events(times: list[float], armed_values: list[float]) -> list[tuple[float, str, str]]:
	if len(times) != len(armed_values):
		raise ValueError('time and armed series must have the same length')
	if not times:
		return []

	events: list[tuple[float, str, str]] = []
	prev_armed = bool(round(armed_values[0]))
	for idx in range(1, len(armed_values)):
		current_armed = bool(round(armed_values[idx]))
		if current_armed != prev_armed:
			label = 'ARM' if current_armed else 'DISARM'
			color = 'tab:green' if current_armed else 'tab:red'
			events.append((times[idx], label, color))
			prev_armed = current_armed
	return events


def find_trace_file(path: Path) -> Path:
	if path.is_file():
		return path
	# Accept both a single default filename and timestamped variants.
	default_trace = path / 'simulation_trace.csv'
	if default_trace.exists():
		return default_trace
	candidates = sorted(path.glob('simulation_trace_*.csv'))
	if not candidates:
		raise FileNotFoundError(f'no simulation_trace.csv or simulation_trace_*.csv found in {path}')
	return candidates[-1]


def resolve_input_path(input_path: str | None, data_folder: str) -> Path:
	if input_path:
		return resolve_path_candidate(Path(input_path))

	data_path = Path(data_folder)
	if len(data_path.parts) > 1:
		return resolve_path_candidate(data_path)
	return SCRIPT_DIR / 'data' / data_folder


def resolve_path_candidate(path: Path) -> Path:
	if path.is_absolute():
		return path

	candidates = [
		Path.cwd() / path,
		SCRIPT_DIR / path,
		REPO_ROOT / path,
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return candidates[0]


def resolve_output_path(out_arg: str | None, input_path: Path) -> Path:
	if not out_arg:
		folder_name = input_path.name if input_path.is_dir() else input_path.parent.name
		return PLOTS_DIR / f'{folder_name}.png'

	out_path = Path(out_arg)
	if out_path.is_absolute():
		return out_path
	return PLOTS_DIR / out_path


def find_params_file(trace_file: Path) -> Path:
	if trace_file.name == 'simulation_trace.csv':
		params_name = 'simulation_params.csv'
	else:
		params_name = trace_file.name.replace('simulation_trace_', 'simulation_params_', 1)
	params_file = trace_file.with_name(params_name)
	if params_file.exists():
		return params_file

	default_params = trace_file.with_name('simulation_params.csv')
	if default_params.exists():
		return default_params

	candidates = sorted(trace_file.parent.glob('simulation_params_*.csv'))
	if not candidates:
		raise FileNotFoundError(
			f'no simulation_params.csv or simulation_params_*.csv found in {trace_file.parent}'
		)
	return candidates[-1]


def load_trace(trace_file: Path) -> dict[str, list[float] | list[str]]:
	with trace_file.open(newline='') as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f'CSV file has no header: {trace_file}')

		required_columns = NUMERIC_COLUMNS + TEXT_COLUMNS
		missing = [col for col in required_columns if col not in reader.fieldnames]
		if missing:
			missing_cols = ', '.join(f'"{col}"' for col in missing)
			raise KeyError(f'expected column(s) {missing_cols} in CSV')

		data: dict[str, list[float] | list[str]] = {
			**{col: [] for col in NUMERIC_COLUMNS},
			**{col: [] for col in TEXT_COLUMNS},
			**{col: [] for col in OPTIONAL_NUMERIC_COLUMNS if col in reader.fieldnames},
		}
		for row_num, row in enumerate(reader, start=2):
			for col in NUMERIC_COLUMNS:
				value = row.get(col)
				if value is None or value == '':
					raise ValueError(f'missing value for "{col}" on row {row_num}')
				try:
					data[col].append(float(value))
				except ValueError as exc:
					raise ValueError(
						f'invalid numeric value for "{col}" on row {row_num}: {value!r}'
					) from exc
			for col in OPTIONAL_NUMERIC_COLUMNS:
				if col not in data:
					continue
				value = row.get(col)
				if value is None or value == '':
					raise ValueError(f'missing value for "{col}" on row {row_num}')
				try:
					data[col].append(float(value))
				except ValueError as exc:
					raise ValueError(
						f'invalid numeric value for "{col}" on row {row_num}: {value!r}'
					) from exc
			for col in TEXT_COLUMNS:
				value = row.get(col)
				if value is None or value == '':
					raise ValueError(f'missing value for "{col}" on row {row_num}')
				data[col].append(value)

	if not data['time_s']:
		raise ValueError(f'CSV file has no data rows: {trace_file}')

	return data


def load_plot_parameters(trace_file: Path) -> dict[str, str]:
	params_file = find_params_file(trace_file)
	with params_file.open(newline='') as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f'CSV file has no header: {params_file}')
		if 'parameter' not in reader.fieldnames or 'value' not in reader.fieldnames:
			raise KeyError(f'expected columns "parameter" and "value" in {params_file}')

		params: dict[str, str] = {}
		for row_num, row in enumerate(reader, start=2):
			parameter = row.get('parameter')
			if parameter in PLOT_PARAMETER_NAMES:
				value = row.get('value')
				if value is None or value == '':
					raise ValueError(f'missing value for "{parameter}" on row {row_num}')
				params[parameter] = value

	missing = [name for name in PLOT_PARAMETER_NAMES if name not in params]
	if missing:
		missing_params = ', '.join(f'"{name}"' for name in missing)
		raise KeyError(f'expected parameter(s) {missing_params} in {params_file}')

	return params


def plot_deltas(
	trace: dict[str, list[float] | list[str]],
	plot_parameters: dict[str, str],
	out_path: Path | None = None,
	show: bool = False,
) -> None:
	t = trace['time_s']
	phases = trace['phase']
	mpc_horizon = plot_parameters['mpc_horizon']
	mpc_q_diag = plot_parameters['mpc_q_diag']
	mpc_qf_diag = plot_parameters['mpc_qf_diag']
	dx = [drone_x - ground_x for drone_x, ground_x in zip(trace['drone_x'], trace['ground_x'])]
	dy = [drone_y - ground_y for drone_y, ground_y in zip(trace['drone_y'], trace['ground_y'])]
	segments = collect_phase_segments(t, phases)
	arm_events = collect_arm_events(t, trace['armed']) if 'armed' in trace else []
	phase_names = list(dict.fromkeys(phases))
	phase_colors = {phase: plt.cm.Set3(idx % plt.cm.Set3.N) for idx, phase in enumerate(phase_names)}

	fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
	for axis in axes:
		for start, end, phase in segments:
			axis.axvspan(start, end, color=phase_colors[phase], alpha=0.2, linewidth=0)
		for event_time, _, color in arm_events:
			axis.axvline(event_time, color=color, linestyle='--', linewidth=1.2, alpha=0.9)

	axes[0].plot(t, dx, label='delta x (drone - ground)')
	axes[0].set_ylabel('delta x (m)')
	axes[0].grid(True)
	axes[0].legend()
	for start, end, phase in segments:
		label_x = start + (end - start) / 2
		axes[0].text(
			label_x,
			1.02,
			phase,
			transform=axes[0].get_xaxis_transform(),
			ha='center',
			va='bottom',
			fontsize=9,
			bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none', 'pad': 1.5},
		)
	for event_time, label, color in arm_events:
		axes[0].text(
			event_time,
			0.94,
			label,
			transform=axes[0].get_xaxis_transform(),
			rotation=90,
			ha='right',
			va='top',
			fontsize=8,
			color=color,
			bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': color, 'pad': 1.0},
		)

	axes[1].plot(t, dy, label='delta y (drone - ground)', color='tab:orange')
	axes[1].set_xlabel('time (s)')
	axes[1].set_ylabel('delta y (m)')
	axes[1].grid(True)
	axes[1].legend()

	fig.suptitle(f'Position deltas vs time (MPC horizon: {mpc_horizon})')
	fig.text(
		0.5,
		0.055,
		f'running weight [px,vx,py,vy,pz,vz]: {mpc_q_diag}',
		ha='center',
		va='bottom',
		fontsize=9,
	)
	fig.text(
		0.5,
		0.025,
		f'terminal weight [px,vx,py,vy,pz,vz]: {mpc_qf_diag}',
		ha='center',
		va='bottom',
		fontsize=9,
	)
	fig.tight_layout(rect=[0, 0.1, 1, 0.94])

	if out_path:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=150)
		print(f'saved plot to {out_path}', file=sys.stderr)
	if show:
		plt.show()

# select the data folder to use when
def main(data_folder="data2"):
	p = argparse.ArgumentParser(description='Plot delta x and delta y from simulation trace')
	p.add_argument('--input', '-i', help='input file or directory containing simulation_trace_*.csv')
	p.add_argument('--data-folder', default=data_folder, help='folder name under data/ to use when --input is not provided')
	p.add_argument('--out', '-o', help='output image path (defaults to sim/plot/<data-folder-name>.png)')
	p.add_argument('--show', action='store_true', help='show plot interactively')
	args = p.parse_args()

	path = resolve_input_path(args.input, args.data_folder)
	if not path.exists():
		print(f'input path does not exist: {path}', file=sys.stderr)
		raise SystemExit(1)

	trace_file = find_trace_file(path)
	trace = load_trace(trace_file)
	plot_parameters = load_plot_parameters(trace_file)
	out_path = resolve_output_path(args.out, path)
	plot_deltas(trace, plot_parameters=plot_parameters, out_path=out_path, show=args.show)


if __name__ == '__main__':
	main('sim/data/run_20260403_161547_583123')
