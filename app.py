import re
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

st.set_page_config(page_title="MedRL Hospital Command Center", layout="wide")

@st.cache_resource
def get_ppo_model():
    try:
        return PPO.load("ppo_hospital_ed_model")
    except Exception:
        return None

@st.cache_data
def load_mimic_samples():
    path = os.path.join("data", "patient_triage.csv")
    if os.path.exists(path):
        return pd.read_csv(path).dropna(subset=["chiefcomplaint"]).head(50)
    return None

ppo_model = get_ppo_model()
mimic_df = load_mimic_samples()

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.doctors = [
        {"id": "DOC-101", "name": "Dr. Aarav Sharma", "specialty": "Emergency & Trauma", "available": True, "current_patient": None},
        {"id": "DOC-102", "name": "Dr. Priya Nair", "specialty": "Critical Care (ICU)", "available": True, "current_patient": None},
        {"id": "DOC-103", "name": "Dr. Rohan Verma", "specialty": "Internal Medicine", "available": True, "current_patient": None},
        {"id": "DOC-104", "name": "Dr. Sneha Kapoor", "specialty": "General Triage", "available": False, "current_patient": None},
    ]
    ward_specs = {
        "ICU Ward": 4,
        "Critical Care Ward": 6,
        "General Ward": 12,
        "Fast-Track / Observation Ward": 8
    }
    beds = []
    for ward, count in ward_specs.items():
        for i in range(1, count + 1):
            beds.append({
                "bed_id": f"{ward[:3].upper()}-{i:02d}",
                "ward": ward,
                "status": "Available",
                "patient_id": "-",
                "patient_name": "-",
                "condition": "-",
                "doctor": "-"
            })
    st.session_state.beds = pd.DataFrame(beds)
    st.session_state.waiting_queue = []
    st.session_state.patient_counter = 1001
    st.session_state.logs = []
    st.session_state.parsed_report = None

def log_event(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{ts}] {msg}")

def analyze_medical_report(report_text, fallback_hr=88, fallback_sbp=120, fallback_spo2=97):
    text_lower = report_text.lower()
    hr_match = re.search(r'(?:hr|heart\s*rate|pulse)\s*[:=\-]?\s*(\d{2,3})', text_lower)
    bp_match = re.search(r'(?:bp|sbp|blood\s*pressure)\s*[:=\-]?\s*(\d{2,3})', text_lower)
    spo2_match = re.search(r'(?:spo2|o2\s*sat|oxygen|o2)\s*[:=\-]?\s*(\d{2,3})', text_lower)

    hr = int(hr_match.group(1)) if hr_match else fallback_hr
    sbp = int(bp_match.group(1)) if bp_match else fallback_sbp
    spo2 = int(spo2_match.group(1)) if spo2_match else fallback_spo2

    icu_keywords = ["cardiac arrest", "myocardial infarction", "acute mi", "stroke", "sepsis", "respiratory failure", "unconscious", "intubat", "severe trauma", "hemorrhage", "coma"]
    crit_keywords = ["chest pain", "dyspnea", "shortness of breath", "troponin", "arrhythmia", "syncope", "pneumonia", "diabetic ketoacidosis", "fracture", "high fever"]
    gen_keywords = ["abdominal pain", "vomiting", "dehydration", "infection", "migraine", "asthma", "moderate pain"]

    flags = []
    for kw in icu_keywords:
        if kw in text_lower:
            flags.append(f"CRITICAL: {kw.upper()}")
    for kw in crit_keywords:
        if kw in text_lower:
            flags.append(f"HIGH RISK: {kw.title()}")
    for kw in gen_keywords:
        if kw in text_lower:
            flags.append(f"MODERATE: {kw.title()}")

    if spo2 < 88 or sbp < 85 or hr > 140 or hr < 45 or any("CRITICAL:" in f for f in flags):
        esi, ward, risk = 1, "ICU Ward", "Critical Life-Threatening (ESI-1)"
    elif spo2 < 92 or sbp < 95 or hr > 120 or any("HIGH RISK:" in f for f in flags):
        esi, ward, risk = 2, "Critical Care Ward", "High-Risk Emergent (ESI-2)"
    elif spo2 < 95 or hr > 100 or any("MODERATE:" in f for f in flags):
        esi, ward, risk = 3, "General Ward", "Urgent Stable (ESI-3)"
    else:
        esi, ward, risk = 4, "Fast-Track / Observation Ward", "Non-Urgent / Minor (ESI-4)"

    if not flags:
        flags.append("Routine Vitals / Standard Clinical Evaluation")

    return {"hr": hr, "sbp": sbp, "spo2": spo2, "esi": esi, "ward": ward, "risk": risk, "flags": flags}

def rl_recommend_action():
    q = st.session_state.waiting_queue
    if not q:
        return None, "Queue empty"
    free_docs = sum(1 for d in st.session_state.doctors if d["available"] and d["current_patient"] is None)
    beds_df = st.session_state.beds
    free_icu = len(beds_df[(beds_df["ward"].isin(["ICU Ward", "Critical Care Ward"])) & (beds_df["status"] == "Available")])
    free_gen = len(beds_df[(beds_df["ward"].isin(["General Ward", "Fast-Track / Observation Ward"])) & (beds_df["status"] == "Available")])

    if ppo_model is not None:
        q_lens = [sum(1 for p in q if p["esi"] == a) for a in range(1, 6)]
        q_waits = [max([p["wait_mins"] for p in q if p["esi"] == a], default=0.0) for a in range(1, 6)]
        obs = np.array(q_lens + q_waits + [free_docs, free_gen, free_icu, 0.5], dtype=np.float32)
        action, _ = ppo_model.predict(obs, deterministic=True)
        target_esi = int(action) + 1
        matching = [p for p in q if p["esi"] == target_esi]
        if matching:
            return matching[0], f"PPO RL Agent selected ESI-{target_esi} patient ({matching[0]['name']}) based on report severity & bed availability."

    sorted_q = sorted(q, key=lambda x: x["esi"])
    return sorted_q[0], f"RL Priority selected highest acuity ESI-{sorted_q[0]['esi']} ({sorted_q[0]['name']})."

def admit_patient_to_ward(patient):
    beds_df = st.session_state.beds
    available_doctors = [d for d in st.session_state.doctors if d["available"] and d["current_patient"] is None]
    if not available_doctors:
        return False, "No free doctor available! Mark a doctor available or add a new doctor."

    target_ward = patient["recommended_ward"]
    free_beds = beds_df[(beds_df["ward"] == target_ward) & (beds_df["status"] == "Available")]
    if free_beds.empty and target_ward != "General Ward":
        free_beds = beds_df[(beds_df["ward"] == "General Ward") & (beds_df["status"] == "Available")]
        if not free_beds.empty:
            target_ward = "General Ward"

    if free_beds.empty:
        return False, f"All beds in {patient['recommended_ward']} and General Ward are full!"

    bed_idx = free_beds.index[0]
    doc = available_doctors[0]
    st.session_state.beds.loc[bed_idx, ["status", "patient_id", "patient_name", "condition", "doctor"]] = [
        "Occupied", patient["id"], patient["name"], f"ESI-{patient['esi']} | {patient['complaint']}", doc["name"]
    ]
    doc["current_patient"] = f"{patient['name']} ({st.session_state.beds.loc[bed_idx, 'bed_id']})"
    st.session_state.waiting_queue = [p for p in st.session_state.waiting_queue if p["id"] != patient["id"]]
    log_event(f"Admitted {patient['name']} (ESI-{patient['esi']}) -> {target_ward} [{st.session_state.beds.loc[bed_idx, 'bed_id']}] under {doc['name']}")
    return True, f"Assigned to {target_ward} ({st.session_state.beds.loc[bed_idx, 'bed_id']}) with {doc['name']}"

st.title("Hospital Emergency & Ward Management System (RL-Powered)")
beds_df = st.session_state.beds
total_beds = len(beds_df)
occ_beds = len(beds_df[beds_df["status"] == "Occupied"])
free_beds = total_beds - occ_beds
free_docs = sum(1 for d in st.session_state.doctors if d["available"] and d["current_patient"] is None)
total_docs = len(st.session_state.doctors)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Available Doctors (Free)", f"{free_docs} / {total_docs}")
m2.metric("Available Beds (Khali)", f"{free_beds} / {total_beds}")
m3.metric("Occupied Beds (Bhare)", f"{occ_beds} / {total_beds}")
m4.metric("Patients in Triage Queue", f"{len(st.session_state.waiting_queue)}")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "1. Wards & Bed List (Khali / Bhare)",
    "2. Doctors Roster (Add / Availability)",
    "3. Patient Report Detector & RL Allocation"
])

with tab1:
    st.subheader("Ward-Wise Bed Summary")
    ward_summary = []
    for w_name, grp in beds_df.groupby("ward", sort=False):
        tot = len(grp)
        occ = len(grp[grp["status"] == "Occupied"])
        ward_summary.append({
            "Ward Name": w_name,
            "Total Beds": tot,
            "Khali (Available)": tot - occ,
            "Bhare (Occupied)": occ,
            "Occupancy %": f"{int((occ/tot)*100)}%"
        })
    st.dataframe(pd.DataFrame(ward_summary), use_container_width=True, hide_index=True)

    st.subheader("Complete Bed List & Discharge Control")
    ward_filter = st.selectbox("Filter by Ward", ["All Wards"] + list(beds_df["ward"].unique()))
    filtered_beds = beds_df if ward_filter == "All Wards" else beds_df[beds_df["ward"] == ward_filter]
    st.dataframe(filtered_beds, use_container_width=True, hide_index=True)

    occupied_rows = beds_df[beds_df["status"] == "Occupied"]
    if not occupied_rows.empty:
        col_d1, col_d2 = st.columns([3, 1])
        bed_to_free = col_d1.selectbox(
            "Select Occupied Bed to Discharge Patient:",
            occupied_rows["bed_id"] + " - " + occupied_rows["patient_name"] + " (" + occupied_rows["ward"] + ")"
        )
        if col_d2.button("Discharge Patient", type="primary"):
            selected_bed_id = bed_to_free.split(" - ")[0]
            row_idx = beds_df[beds_df["bed_id"] == selected_bed_id].index[0]
            doc_name = beds_df.loc[row_idx, "doctor"]
            p_name = beds_df.loc[row_idx, "patient_name"]
            for d in st.session_state.doctors:
                if d["name"] == doc_name:
                    d["current_patient"] = None
            st.session_state.beds.loc[row_idx, ["status", "patient_id", "patient_name", "condition", "doctor"]] = [
                "Available", "-", "-", "-", "-"
            ]
            log_event(f"Discharged {p_name} from {selected_bed_id}. {doc_name} is now FREE.")
            st.rerun()

with tab2:
    col_doc_list, col_doc_add = st.columns([1.6, 1])
    with col_doc_list:
        st.subheader("Current Hospital Doctors")
        for idx, doc in enumerate(st.session_state.doctors):
            c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1.2])
            c1.write(f"**{doc['name']}** ({doc['id']})")
            c2.write(f"{doc['specialty']}")
            if doc["current_patient"]:
                c3.markdown(f":red[Treating: {doc['current_patient']}]")
            elif doc["available"]:
                c3.markdown(":green[Available (On Duty)]")
            else:
                c3.markdown(":orange[Off Duty / Unavailable]")
            toggle_label = "Set Off-Duty" if doc["available"] else "Set Available"
            if c4.button(toggle_label, key=f"doc_toggle_{idx}", disabled=bool(doc["current_patient"])):
                doc["available"] = not doc["available"]
                log_event(f"{doc['name']} status changed to {'Available' if doc['available'] else 'Off-Duty'}.")
                st.rerun()

    with col_doc_add:
        st.subheader("Add New Doctor Manually")
        with st.form("add_doctor_form", clear_on_submit=True):
            new_doc_name = st.text_input("Doctor Name", placeholder="e.g. Dr. Kabir Mehta")
            new_doc_spec = st.selectbox("Department / Specialty", [
                "Emergency & Trauma", "Critical Care (ICU)", "Cardiology", "General Medicine", "Surgery"
            ])
            new_doc_avail = st.checkbox("Available Right Now", value=True)
            submitted = st.form_submit_button("Add Doctor to Hospital", type="primary")
            if submitted and new_doc_name.strip():
                new_id = f"DOC-{101 + len(st.session_state.doctors)}"
                st.session_state.doctors.append({
                    "id": new_id,
                    "name": new_doc_name.strip(),
                    "specialty": new_doc_spec,
                    "available": new_doc_avail,
                    "current_patient": None
                })
                log_event(f"Added new doctor: {new_doc_name.strip()} ({new_doc_spec})")
                st.rerun()

with tab3:
    col_rep, col_q = st.columns([1.15, 1.25])

    with col_rep:
        st.subheader("Upload / Analyze Patient Medical Report")
        if mimic_df is not None:
            if st.button("Load Random Real MIMIC-IV-ED Patient Report"):
                sample = mimic_df.sample(1).iloc[0]
                hr_v = int(sample["heartrate"]) if pd.notna(sample["heartrate"]) else 90
                sbp_v = int(sample["sbp"]) if pd.notna(sample["sbp"]) else 120
                spo2_v = int(sample["o2sat"]) if pd.notna(sample["o2sat"]) else 97
                rep_str = (
                    f"Patient Stay ID: {int(sample['stay_id'])} | Gender: {sample['gender']} | "
                    f"Transport: {sample['arrival_transport']}\n"
                    f"Chief Complaint: {sample['chiefcomplaint']}\n"
                    f"Vitals: HR: {hr_v} bpm, BP: {sbp_v} mmHg, SpO2: {spo2_v}%, Temp: {sample['temperature']} F"
                )
                st.session_state.sample_name = f"MIMIC-PT-{int(sample['subject_id'])}"
                st.session_state.sample_report = rep_str

        uploaded_file = st.file_uploader("Upload Patient Report File (.txt or .csv)", type=["txt", "csv"])
        file_content = ""
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")

        p_name = st.text_input(
            "Patient Name / ID",
            value=st.session_state.get("sample_name", ""),
            placeholder="e.g. Amit Verma"
        )
        report_text = st.text_area(
            "Medical / Lab Report Notes (Paste or Upload above)",
            value=file_content if file_content else st.session_state.get("sample_report", ""),
            height=130,
            placeholder="Example: Patient reports severe chest pain and shortness of breath. HR: 138, BP: 84, SpO2: 86%, Troponin elevated."
        )

        with st.expander("Manual Vitals Override (Optional if not written in report)"):
            c_v1, c_v2, c_v3 = st.columns(3)
            m_hr = c_v1.number_input("Heart Rate (bpm)", 30, 220, 88)
            m_sbp = c_v2.number_input("Systolic BP (mmHg)", 50, 240, 120)
            m_spo2 = c_v3.number_input("SpO2 Oxygen (%)", 60, 100, 97)

        if st.button("1. Analyze Report & Detect Ward", type="secondary"):
            if not report_text.strip():
                st.warning("Please upload a report file or enter report text first.")
            else:
                analysis = analyze_medical_report(report_text, m_hr, m_sbp, m_spo2)
                analysis["patient_name"] = p_name.strip() or f"Patient-{st.session_state.patient_counter}"
                st.session_state.parsed_report = analysis

        if st.session_state.parsed_report is not None:
            res = st.session_state.parsed_report
            st.success(f"**Detected Ward:** {res['ward']}  |  **Severity:** {res['risk']}")
            st.write(f"**Extracted Vitals:** HR `{res['hr']} bpm` | BP `{res['sbp']} mmHg` | SpO2 `{res['spo2']}%`")
            st.write("**Clinical Flags Detected:** " + " • ".join(res["flags"]))

            c_b1, c_b2 = st.columns(2)
            if c_b1.button("2. Admit Now (Auto RL Ward + Doctor)", type="primary"):
                new_patient = {
                    "id": f"PT-{st.session_state.patient_counter}",
                    "name": res["patient_name"],
                    "complaint": ", ".join(res["flags"])[:55],
                    "vitals": f"HR:{res['hr']} | BP:{res['sbp']} | SpO2:{res['spo2']}%",
                    "esi": res["esi"],
                    "risk": res["risk"],
                    "recommended_ward": res["ward"],
                    "wait_mins": 0
                }
                st.session_state.patient_counter += 1
                st.session_state.waiting_queue.append(new_patient)
                ok, msg = admit_patient_to_ward(new_patient)
                st.session_state.parsed_report = None
                st.session_state.sample_name = ""
                st.session_state.sample_report = ""
                if ok:
                    st.success(msg)
                else:
                    st.warning(f"Added to queue: {msg}")
                st.rerun()

            if c_b2.button("Add to Waiting Queue Only"):
                new_patient = {
                    "id": f"PT-{st.session_state.patient_counter}",
                    "name": res["patient_name"],
                    "complaint": ", ".join(res["flags"])[:55],
                    "vitals": f"HR:{res['hr']} | BP:{res['sbp']} | SpO2:{res['spo2']}%",
                    "esi": res["esi"],
                    "risk": res["risk"],
                    "recommended_ward": res["ward"],
                    "wait_mins": 0
                }
                st.session_state.patient_counter += 1
                st.session_state.waiting_queue.append(new_patient)
                log_event(f"Queued {new_patient['name']} -> Detected {res['ward']} (ESI-{res['esi']})")
                st.session_state.parsed_report = None
                st.rerun()

    with col_q:
        st.subheader("Triage Waiting Queue & PPO RL Allocation")
        if st.session_state.waiting_queue:
            rec_patient, reason = rl_recommend_action()
            st.info(f"**RL Agent Recommendation:** {reason}")
            if st.button("Execute RL Recommendation (Admit Next Patient)", type="primary"):
                ok, msg = admit_patient_to_ward(rec_patient)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            q_df = pd.DataFrame(st.session_state.waiting_queue)[
                ["id", "name", "esi", "vitals", "recommended_ward", "complaint"]
            ]
            st.dataframe(q_df, use_container_width=True, hide_index=True)
        else:
            st.write("No patients currently waiting in queue.")

st.divider()
st.subheader("Hospital Activity Log")
for entry in st.session_state.logs[:8]:
    st.text(entry)