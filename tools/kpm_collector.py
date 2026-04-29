"""KPM 데이터 수집기 — CSV 출력"""
import time, sys, os, csv
sys.path.insert(0, '/usr/local/lib/python3/dist-packages/xapp_sdk/')
import xapp_sdk as ric

METRICS = ['DRB.UEThpDl', 'DRB.UEThpUl', 'RRU.PrbTotDl', 'RRU.PrbTotUl',
           'CQI', 'RSRP', 'DL.BLER', 'UL.BLER', 'PUCCH.SINR']

class KPMCollector(ric.kpm_cb):
    def __init__(self, label, csv_writer, duration):
        ric.kpm_cb.__init__(self)
        self.label = label
        self.writer = csv_writer
        self.count = 0
        self.duration = duration
        self.start_time = time.time()

    def handle(self, ind):
        self.count += 1
        if ind.msg.type != ric.FORMAT_1_INDICATION_MESSAGE:
            return
        frm1 = ind.msg.frm_1
        row = {'timestamp': time.time(), 'label': self.label}
        for meas_data in frm1.meas_data_lst:
            for j, rec in enumerate(meas_data.meas_record_lst):
                if j < len(frm1.meas_info_lst):
                    info = frm1.meas_info_lst[j]
                    name = info.meas_type.name if info.meas_type.type == ric.NAME_MEAS_TYPE else str(info.meas_type.id)
                else:
                    name = f'm{j}'
                if rec.value == ric.REAL_MEAS_VALUE:
                    row[name] = rec.real_val
                elif rec.value == ric.INTEGER_MEAS_VALUE:
                    row[name] = rec.int_val
                else:
                    row[name] = 0
        self.writer.writerow(row)
        elapsed = time.time() - self.start_time
        if self.count % 5 == 0:
            vals = {k: row.get(k, '?') for k in ['CQI', 'PUCCH.SINR', 'DL.BLER', 'DRB.UEThpDl']}
            print(f'  [{self.label}] #{self.count} ({elapsed:.0f}s) {vals}', flush=True)

if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else 'Normal'
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    outfile = sys.argv[3] if len(sys.argv) > 3 else '/tmp/kpm_data.csv'

    file_exists = os.path.exists(outfile) and os.path.getsize(outfile) > 0
    f = open(outfile, 'a', newline='')
    fieldnames = ['timestamp', 'label'] + METRICS
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    if not file_exists:
        writer.writeheader()

    ric.init([sys.argv[0], '-c', '/tmp/xapp_kpm_hello.conf'])
    conn = ric.conn_e2_nodes()
    print(f'E2 nodes: {len(conn)}', flush=True)
    if len(conn) == 0:
        print('No E2 nodes!', flush=True)
        os._exit(1)

    cb = KPMCollector(label, writer, duration)
    h = ric.report_kpm_sm(conn[0].id, ric.Interval_ms_1000, METRICS, cb)
    print(f'Collecting "{label}" for {duration}s...', flush=True)

    time.sleep(duration)
    f.close()
    print(f'Done: {cb.count} samples → {outfile}', flush=True)
    os._exit(0)
