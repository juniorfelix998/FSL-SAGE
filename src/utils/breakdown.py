# ------------------------------------------------------------------------------
# Per-component communication & memory breakdown, labelled by the paper step
# each transfer comes from.
#
# WHY: the benchmark table gives one number per column (Cut, Weights, ...). That
# number is a sum of several protocol steps -- smashed-data uploads, labels,
# client FedAvg, auxiliary-model transfers -- and a reader cannot check it
# against the paper without knowing which step contributed what. Every row here
# names the step, cites where the paper defines it, and shows
#     count  x  size per transfer  =  MiB
# next to the count the paper's schedule predicts, so a ✗ means our run did not
# follow the paper.
#
# Units are MiB (1024**2 bytes), the same unit as the benchmark table's "MB"
# columns.
#
# Only CSE-FSL and FSL-SAGE have paper-labelled rows so far; any other method
# falls back to the raw ledger categories.
# ------------------------------------------------------------------------------
import math

MB = 1024 ** 2

SERVER_FEDAVG_NAME = 'Server-model FedAvg'
SERVER_FEDAVG_HOW = (
    "part of Weights spent averaging per-client copies of the SERVER-side "
    "model = weights.server_up + weights.server_down = N clients x R rounds x "
    "2 (up+down) x |x_s|. Already inside Weights, not extra. 0 for any method "
    "that keeps a single server model."
)


# -- expected-count schedules ----------------------------------------------------
# `s` is the schedule dict produced by FLAlgorithm.schedule().
def _uploads_every_h(s):
    '''Smashed-data uploads on the LAST batch of every h-batch window (paper
    batch k = h, 2h, ... <= K): floor(K/h) per client per round.
      CSE-FSL : Eq.(8) / Fig.2 -- server gets g_{x_c^{t,h-1}} "when m = h"
      FSL-SAGE: Alg.1 l.11 / Alg.3 l.3 -- k in [K], k = 0 mod K/Q'''
    h = s['h']
    per_round = sum(
        (e * it) // h for e, it in zip(s['epochs'], s['iters_per_epoch'])
    )
    return per_round * s['rounds']


def _per_client_per_round(s):
    return s['num_clients'] * s['rounds']


def _initial_plus_per_round(s):
    '''FSL-SAGE Alg.2: broadcast x_c^0 (l.4) + x_c^{t+1} after every round (l.8).'''
    return s['num_clients'] * (s['rounds'] + 1)


def _aux_down_every_l(s):
    '''FSL-SAGE: aligned aux model sent to each client in rounds t % l == 0.'''
    return s['num_clients'] * math.ceil(s['rounds'] / s['l'])


def _never(s):
    return 0


# -- paper-labelled rows ---------------------------------------------------------
# (id, what is sent, paper reference, ledger key(s), expected-count fn,
#  table column it lands in)
PAPER_COMPONENTS = {
    'cse_fsl': {
        'paper': 'CSE-FSL, Mu & Shen, arXiv:2302.05599',
        'rows': [
            ('C1', 'Smashed activations z_f, client -> server',
             'Eq.(8), Fig.2 (p.3-4): last batch of each h-batch window (m = h); N*R*floor(K/h)',
             ('cut.act_up',), _uploads_every_h, 'Cut'),
            ('C2', 'Labels y for each uploaded batch, client -> server',
             'Step 2 (p.3): "uploads the smashed data and the labels"; N*R*floor(K/h)',
             ('cut.labels_up',), _uploads_every_h, 'Cut'),
            ('C3', 'Gradient at the cut, server -> client',
             'p.3: never sent; client trains on its local aux loss',
             ('cut.grad_down',), _never, 'Cut'),
            ('W1', 'Client model x_c upload (for FedAvg)',
             'Step 4, Eq.(9) (p.4): every round; N*R',
             ('weights.client_up',), _per_client_per_round, 'Weights'),
            ('W2', 'Averaged client model x_c download',
             'Step 1 (p.3): start of every round; N*R',
             ('weights.client_down',), _per_client_per_round, 'Weights'),
            ('W3', 'Aux model a_c upload (for FedAvg)',
             'Step 4, Eq.(9) (p.4): aux is averaged every round; N*R',
             ('weights.aux_up',), _per_client_per_round, 'Weights'),
            ('W4', 'Averaged aux model a_c download',
             'Step 1 (p.3): start of every round; N*R',
             ('weights.aux_down',), _per_client_per_round, 'Weights'),
            ('W5', f'{SERVER_FEDAVG_NAME} (per-client x_s copies)',
             'p.3: "keep only a single server-side model" -> none',
             ('weights.server_up', 'weights.server_down'), _never, 'Weights'),
        ],
    },
    'fsl_sage': {
        'paper': 'FSL-SAGE, Nair et al., arXiv:2505.23182',
        'rows': [
            ('C1', 'Smashed activations z_f, client -> server',
             'Alg.1 l.11-12 / Alg.3 l.3: k in [K], k = 0 mod K/Q (K/Q = h); N*R*floor(K/h)',
             ('cut.act_up',), _uploads_every_h, 'Cut'),
            ('C2', 'Labels y for each uploaded batch, client -> server',
             'Alg.3 l.3-4: server receives (z_f, y); N*R*floor(K/h)',
             ('cut.labels_up',), _uploads_every_h, 'Cut'),
            ('C3', 'Gradient at the cut, server -> client',
             'Alg.3 l.11-12: true grads used only on the server -> never sent',
             ('cut.grad_down',), _never, 'Cut'),
            ('W1', 'Client model x_c upload (for FedAvg)',
             'Alg.2 l.6: every round; N*R',
             ('weights.client_up',), _per_client_per_round, 'Weights'),
            ('W2', 'Client model x_c download (initial + averaged)',
             'Alg.2 l.4 (x_c^0) + l.8 (after every round); N*(R+1)',
             ('weights.client_down',), _initial_plus_per_round, 'Weights'),
            ('W3', 'Aux model x_a upload',
             'Alg.2: aux models are never averaged -> never sent up',
             ('weights.aux_up',), _never, 'Weights'),
            ('W4', 'Aligned aux model x_a download, server -> client i',
             'Alg.1 l.3-4, Fig.1: start of every round t = 0 mod l; N*ceil(R/l)',
             ('weights.aux_down',), _aux_down_every_l, 'Weights'),
            ('W5', f'{SERVER_FEDAVG_NAME} (per-client x_s copies)',
             'p.8: "a single server-side model is maintained" -> none',
             ('weights.server_up', 'weights.server_down'), _never, 'Weights'),
        ],
    },
}


# ------------------------------------------------------------------------------
def build_breakdown(algo_name, schedule, snapshot, totals, memory_metrics):
    '''Pure-data breakdown (JSON-serialisable) for one run.

    snapshot       -- CommLedger.snapshot() at the end of the run
    totals         -- {'cut': bytes, 'weights': bytes, 'total': bytes}
    memory_metrics -- FLAlgorithm.memory_report()
    '''
    spec = PAPER_COMPONENTS.get(algo_name)
    rows, covered = [], set()

    if spec is not None:
        for rid, what, ref, keys, expect_fn, column in spec['rows']:
            nbytes = sum(snapshot.get(k, 0.0) for k in keys)
            count = sum(snapshot.get(f'count.{k}', 0) for k in keys)
            expected = expect_fn(schedule)
            covered.update(keys)
            rows.append({
                'id': rid, 'item': what, 'paper_ref': ref, 'column': column,
                'ledger_keys': list(keys), 'count': count,
                'expected_count': expected,
                'mib_per_transfer': (nbytes / count / MB) if count else 0.0,
                'mib': nbytes / MB, 'ok': count == expected,
            })

    # anything the ledger charged that no paper row explains -- always a ✗
    # for a paper-labelled method, plain listing for the others
    for key, nbytes in snapshot.items():
        if key.startswith('count.') or key in covered or not nbytes:
            continue
        count = snapshot.get(f'count.{key}', 0)
        rows.append({
            'id': '?', 'item': f'{key} (not in the paper protocol)'
            if spec else key,
            'paper_ref': '' , 'column': 'Cut' if key.startswith('cut.')
            else 'Weights', 'ledger_keys': [key], 'count': count,
            'expected_count': 0 if spec else None,
            'mib_per_transfer': (nbytes / count / MB) if count else 0.0,
            'mib': nbytes / MB, 'ok': (False if spec else None),
        })

    server_fedavg = (snapshot.get('weights.server_up', 0.0)
                     + snapshot.get('weights.server_down', 0.0)) / MB
    cut_sum = sum(r['mib'] for r in rows if r['column'] == 'Cut')
    w_sum = sum(r['mib'] for r in rows if r['column'] == 'Weights')

    mm = memory_metrics
    client_lines = mm.get('client_mem_components', [])
    server_lines = mm.get('server_mem_components', [])

    return {
        'algorithm': algo_name,
        'paper': spec['paper'] if spec else None,
        'schedule': schedule,
        'comm_rows': rows,
        'comm_totals_mib': {
            'cut': totals['cut'] / MB, 'weights': totals['weights'] / MB,
            'server_model_fedavg': server_fedavg, 'total': totals['total'] / MB,
            'cut_rows_sum': cut_sum, 'weights_rows_sum': w_sum,
        },
        'server_model_fedavg_how': SERVER_FEDAVG_HOW,
        'memory': {
            'client_lines': client_lines,
            'client_index': mm.get('client_mem_components_client'),
            'client_lines_sum': sum(l['mb'] for l in client_lines),
            'client_total': mm.get('peak_client_mem_mb'),
            'held_across_cut': mm.get('client_mem_held_across_cut_mb'),
            'server_lines': server_lines,
            'server_lines_sum': sum(l['mb'] for l in server_lines),
            'server_total': mm.get('peak_server_mem_mb'),
            'server_extra_lines': mm.get('server_mem_extra_components', []),
            'system_peak_live': mm.get('peak_system_live_mb'),
            'system_peak_with_static': mm.get('peak_system_mem_mb'),
        },
    }


# ------------------------------------------------------------------------------
def _tick(ok):
    return '' if ok is None else ('✓' if ok else '✗')


def _same(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol * max(1, abs(b))


def render(bd):
    '''Human-readable markdown for one run's breakdown.'''
    s = bd['schedule']
    out = []
    title = bd['algorithm'] + (f"  —  {bd['paper']}" if bd['paper'] else '')
    out.append(f"## Breakdown: {title}")
    out.append(
        f"N = {s['num_clients']} clients, R = {s['rounds']} rounds, "
        f"local iters/round = {[e * it for e, it in zip(s['epochs'], s['iters_per_epoch'])]}, "
        f"h = {s.get('h')}, l = {s.get('l')}.  Units: MiB (= table 'MB')."
    )
    out.append('')
    out.append('### Communication')
    out.append('✓ = the run\'s count equals the count derived from the paper\'s pseudocode '
               '(formula and line in the Paper column).')
    out.append('')
    out.append('| # | What is sent | Paper | Column | Count | Paper count | MiB / transfer | MiB | |')
    out.append('|---|---|---|---|---:|---:|---:|---:|---|')
    for r in bd['comm_rows']:
        exp = '' if r['expected_count'] is None else r['expected_count']
        out.append(
            f"| {r['id']} | {r['item']} | {r['paper_ref']} | {r['column']} | "
            f"{r['count']} | {exp} | {r['mib_per_transfer']:.4f} | "
            f"{r['mib']:.2f} | {_tick(r['ok'])} |"
        )
    t = bd['comm_totals_mib']
    out.append('')
    out.append(f"- **Cut** = {t['cut']:.2f} MiB  (rows sum {t['cut_rows_sum']:.2f} {_tick(_same(t['cut_rows_sum'], t['cut']))})")
    out.append(f"- **Weights** = {t['weights']:.2f} MiB  (rows sum {t['weights_rows_sum']:.2f} {_tick(_same(t['weights_rows_sum'], t['weights']))})")
    out.append(f"  - of that, **{SERVER_FEDAVG_NAME}** = {t['server_model_fedavg']:.2f} MiB")
    out.append(f"- **Total** = Cut + Weights = {t['total']:.2f} MiB")
    if t['total']:
        out.append(f"- Cut share = {100 * t['cut'] / t['total']:.0f}%")
    out.append('')
    out.append(f"*{SERVER_FEDAVG_NAME}:* {bd['server_model_fedavg_how']}")

    m = bd['memory']
    out.append('')
    out.append(f"### Memory — client (worst client = {m['client_index']})")
    out.append(_mem_table(m['client_lines']))
    out.append(f"\n- **Client total** = {_f(m['client_total'])} MiB  (lines sum {_f(m['client_lines_sum'])} {_tick(_same(m['client_lines_sum'], m['client_total']))})")
    out.append(f"- **Held-across-cut** = {_f(m['held_across_cut'])} MiB: client autograd graph still alive while the server works (a stall), already inside the working-memory line")
    out.append('')
    out.append('### Memory — server')
    out.append(_mem_table(m['server_lines']))
    out.append(f"\n- **Server total (table value)** = {_f(m['server_total'])} MiB  (lines sum {_f(m['server_lines_sum'])} {_tick(_same(m['server_lines_sum'], m['server_total']))})")
    if m['server_extra_lines']:
        out.append('- Also kept on the server by the protocol, **not** in the Server total (table value):')
        for l in m['server_extra_lines']:
            out.append(f"  - {l['item']}: {l['mb']:.2f} MiB  ({l['note']})")
        full = (m['server_total'] or 0) + sum(l['mb'] for l in m['server_extra_lines'])
        out.append(f"- **Total server memory incl. all clients' state** = {full:.2f} MiB "
                   f"(Server total + lines above; the table uses the per-device Server total)")
    out.append('')
    out.append(f"- **System peak** (live bytes, both sides at one instant) = {_f(m['system_peak_live'])} MiB; "
               f"with static model/optimizer bytes = {_f(m['system_peak_with_static'])} MiB")
    return '\n'.join(out)


def _f(v):
    return 'n/a' if v is None else f'{v:.2f}'


def _mem_table(lines):
    rows = ['| Item | params+buffers | grads | optimizer | MiB |',
            '|---|---:|---:|---:|---:|']
    for l in lines:
        if 'params_mb' in l:
            rows.append(f"| {l['item']} | {l['params_mb']:.2f} | {l['grads_mb']:.2f} | "
                        f"{l['optim_mb']:.2f} | {l['mb']:.2f} |")
        else:
            act = l.get('of_which_activations_mb')
            note = f" (autograd activations at their own peak: {act:.2f})" if act is not None else ''
            rows.append(f"| {l['item']}{note} | | | | {l['mb']:.2f} |")
    return '\n'.join(rows)


def render_side_by_side(bds):
    '''Compact comparison of several runs' paper rows (e.g. CSE-FSL vs FSL-SAGE).'''
    names = [b['algorithm'] for b in bds]
    ids = []
    for b in bds:
        for r in b['comm_rows']:
            if r['id'] not in ids:
                ids.append(r['id'])
    out = ['## Side by side (MiB)', '',
           '| # | ' + ' | '.join(names) + ' |',
           '|---|' + '---:|' * len(names)]
    for rid in ids:
        cells = []
        for b in bds:
            r = next((r for r in b['comm_rows'] if r['id'] == rid), None)
            cells.append('' if r is None else f"{r['mib']:.2f} ({r['count']}x)")
        label = next(r['item'] for b in bds for r in b['comm_rows'] if r['id'] == rid)
        out.append(f"| {rid} {label} | " + ' | '.join(cells) + ' |')
    for key, label in (('cut', 'Cut'), ('weights', 'Weights'),
                       ('server_model_fedavg', SERVER_FEDAVG_NAME),
                       ('total', 'Total')):
        out.append(f"| **{label}** | " + ' | '.join(
            f"{b['comm_totals_mib'][key]:.2f}" for b in bds) + ' |')
    for key, label in (('client_total', 'Client mem'),
                       ('held_across_cut', 'Held-across-cut'),
                       ('server_total', 'Server mem'),
                       ('system_peak_live', 'System peak')):
        out.append(f"| **{label}** | " + ' | '.join(
            _f(b['memory'][key]) for b in bds) + ' |')
    return '\n'.join(out)
