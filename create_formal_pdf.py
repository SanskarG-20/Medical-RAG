"""
Generate a Multi-Page Formal Clinical Diagnostic Report PDF for MedRAG-X Testing.
Properly formats hospital banners, patient metadata, structured lab tables, and clinical impressions.
"""
import fitz

def create_formal_clinical_pdf(output_path="Formal-Clinical-Patient-Report.pdf"):
    doc = fitz.open()
    
    NAVY = (0.05, 0.15, 0.35)
    LIGHT_GRAY = (0.95, 0.96, 0.98)
    BORDER_COLOR = (0.75, 0.78, 0.82)
    TEXT_DARK = (0.1, 0.1, 0.1)
    RED_FLAG = (0.8, 0.1, 0.1)
    
    # ---------------- PAGE 1 ----------------
    page1 = doc.new_page(width=612, height=792)
    
    # Header Banner
    header_rect = fitz.Rect(36, 36, 576, 85)
    page1.draw_rect(header_rect, color=NAVY, fill=NAVY)
    page1.insert_text(fitz.Point(48, 58), "METROPOLIS ADVANCED HEALTHCARE SYSTEM", color=(1, 1, 1), fontsize=13, fontname="helv")
    page1.insert_text(fitz.Point(48, 73), "DEPARTMENT OF PATHOLOGY & CLINICAL LABORATORY MEDICINE", color=(0.8, 0.85, 0.95), fontsize=8, fontname="helv")
    page1.insert_text(fitz.Point(410, 58), "CLIA ID: 99D0874521", color=(1, 1, 1), fontsize=8, fontname="helv")
    page1.insert_text(fitz.Point(410, 70), "CAP ACCREDITED: #1478201", color=(0.8, 0.85, 0.95), fontsize=8, fontname="helv")
    
    # Patient Demographics Box
    demo_rect = fitz.Rect(36, 95, 576, 175)
    page1.draw_rect(demo_rect, color=BORDER_COLOR, fill=LIGHT_GRAY, width=1)
    
    page1.insert_text(fitz.Point(46, 115), "Patient Name: Eleanor Vance", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(46, 132), "Patient ID / MRN: PT-90482", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(46, 149), "DOB: 1972-04-12 (Age: 54 Yrs)", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(46, 166), "Gender: Female", color=TEXT_DARK, fontsize=9, fontname="helv")
    
    page1.insert_text(fitz.Point(230, 115), "Accession #: ACC-2026-88491", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(230, 132), "Collection Date: 2026-10-25 08:30 AM", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(230, 149), "Received Date: 2026-10-25 09:15 AM", color=TEXT_DARK, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(230, 166), "Report Date: 2026-10-26 14:00 PM", color=TEXT_DARK, fontsize=9, fontname="helv")
    
    page1.insert_text(fitz.Point(420, 115), "Ordering Physician:", color=TEXT_DARK, fontsize=8, fontname="helv")
    page1.insert_text(fitz.Point(420, 128), "Dr. Marcus Vance, MD", color=NAVY, fontsize=9, fontname="helv")
    page1.insert_text(fitz.Point(420, 145), "Specimen: Whole Blood EDTA", color=TEXT_DARK, fontsize=8, fontname="helv")
    page1.insert_text(fitz.Point(420, 162), "Status: FINAL VERIFIED", color=(0.1, 0.5, 0.2), fontsize=8, fontname="helv")
    
    # Clinical Presentation
    page1.insert_text(fitz.Point(36, 195), "CLINICAL PRESENTATION & REASON FOR TESTING", color=NAVY, fontsize=10, fontname="helv")
    hist_lines = [
        "Patient presents with a 4-month history of severe fatigue, exertional dyspnea, restless leg syndrome, and cold intolerance.",
        "Physical examination notable for marked conjunctival pallor, koilonychia, and resting tachycardia (HR 102 bpm).",
        "Patient reports ongoing heavy menorrhagia over the preceding 6 months with suspected chronic blood loss.",
        "Ordering evaluation for microcytic hypochromic anemia and iron deficiency."
    ]
    cur_y = 212
    for line in hist_lines:
        page1.insert_text(fitz.Point(40, cur_y), line, color=TEXT_DARK, fontsize=8.5, fontname="helv")
        cur_y += 14
        
    def draw_table(target_page, start_y, title, headers, rows):
        target_page.insert_text(fitz.Point(36, start_y), title, color=NAVY, fontsize=10, fontname="helv")
        table_top = start_y + 8
        row_height = 18
        col_widths = [190, 80, 80, 110, 80]
        total_width = sum(col_widths)
        
        # Header
        hdr_rect = fitz.Rect(36, table_top, 36 + total_width, table_top + row_height)
        target_page.draw_rect(hdr_rect, color=NAVY, fill=NAVY)
        
        cur_x = 36
        for i, h in enumerate(headers):
            target_page.insert_text(fitz.Point(cur_x + 6, table_top + 13), h, color=(1, 1, 1), fontsize=8.5, fontname="helv")
            cur_x += col_widths[i]
            
        cur_y = table_top + row_height
        for r_idx, r in enumerate(rows):
            fill_c = (1, 1, 1) if r_idx % 2 == 0 else (0.96, 0.97, 0.99)
            row_rect = fitz.Rect(36, cur_y, 36 + total_width, cur_y + row_height)
            target_page.draw_rect(row_rect, color=BORDER_COLOR, fill=fill_c, width=0.5)
            
            cur_x = 36
            for c_idx, cell in enumerate(r):
                is_flag = (c_idx == 4 and cell in ["CRITICAL", "LOW", "HIGH"])
                text_col = RED_FLAG if is_flag else TEXT_DARK
                target_page.insert_text(fitz.Point(cur_x + 6, cur_y + 13), str(cell), color=text_col, fontsize=8.5, fontname="helv")
                cur_x += col_widths[c_idx]
            cur_y += row_height
            
        return cur_y + 18

    # Table 1: CBC
    cbc_headers = ["Test Name", "Result", "Units", "Reference Range", "Flag"]
    cbc_rows = [
        ["Hemoglobin (Hb)", "7.8", "g/dL", "12.0 - 15.5", "CRITICAL"],
        ["Hematocrit (HCT)", "24.6", "%", "37.0 - 48.0", "LOW"],
        ["RBC Count", "2.95", "x10^6/uL", "3.80 - 5.20", "LOW"],
        ["Mean Corpuscular Volume (MCV)", "68.2", "fL", "80.0 - 100.0", "LOW"],
        ["Mean Corpuscular Hemoglobin (MCH)", "20.4", "pg", "27.0 - 33.0", "LOW"],
        ["MCHC", "29.5", "g/dL", "32.0 - 36.0", "LOW"],
        ["Red Cell Distribution Width (RDW)", "19.4", "%", "11.5 - 14.5", "HIGH"],
        ["White Blood Cell (WBC) Count", "6.4", "x10^3/uL", "4.5 - 11.0", "NORMAL"],
        ["Platelet Count", "445", "x10^3/uL", "150 - 400", "HIGH"]
    ]
    next_y = draw_table(page1, cur_y + 10, "1. COMPLETE BLOOD COUNT (CBC) & RED CELL MORPHOLOGY", cbc_headers, cbc_rows)
    
    # Table 2: Iron Profile
    iron_headers = ["Test Name", "Result", "Units", "Reference Range", "Flag"]
    iron_rows = [
        ["Serum Ferritin", "4.8", "ng/mL", "15.0 - 150.0", "CRITICAL"],
        ["Serum Iron", "18", "ug/dL", "60 - 170", "LOW"],
        ["Total Iron Binding Capacity (TIBC)", "485", "ug/dL", "240 - 450", "HIGH"],
        ["Transferrin Saturation (TSAT)", "3.7", "%", "20.0 - 50.0", "LOW"]
    ]
    draw_table(page1, next_y, "2. IRON STATUS & BIOCHEMICAL HEMATOLOGY PANEL", iron_headers, iron_rows)
    
    page1.insert_text(fitz.Point(36, 755), "Metropolis Healthcare System | Laboratory Medicine | Confidential Report", color=(0.5, 0.5, 0.5), fontsize=8, fontname="helv")
    page1.insert_text(fitz.Point(500, 755), "Page 1 of 2", color=(0.5, 0.5, 0.5), fontsize=8, fontname="helv")

    # ---------------- PAGE 2 ----------------
    page2 = doc.new_page(width=612, height=792)
    
    # Page 2 Mini Header
    p2_hdr = fitz.Rect(36, 36, 576, 65)
    page2.draw_rect(p2_hdr, color=NAVY, fill=NAVY)
    page2.insert_text(fitz.Point(48, 54), "METROPOLIS ADVANCED HEALTHCARE SYSTEM - PATIENT REPORT CONTINUED", color=(1, 1, 1), fontsize=9.5, fontname="helv")
    page2.insert_text(fitz.Point(450, 54), "MRN: PT-90482", color=(1, 1, 1), fontsize=9.5, fontname="helv")
    
    # Table 3: CMP & Glycemic
    cmp_headers = ["Test Name", "Result", "Units", "Reference Range", "Flag"]
    cmp_rows = [
        ["Glycated Hemoglobin (HbA1c)", "5.3", "%", "< 5.7", "NORMAL"],
        ["Fasting Serum Glucose", "88", "mg/dL", "70 - 99", "NORMAL"],
        ["Serum Creatinine", "0.78", "mg/dL", "0.50 - 1.10", "NORMAL"],
        ["Blood Urea Nitrogen (BUN)", "12", "mg/dL", "7 - 20", "NORMAL"],
        ["Estimated GFR (eGFR)", "> 90", "mL/min/1.73m2", "> 60", "NORMAL"],
        ["Serum Sodium", "140", "mEq/L", "136 - 145", "NORMAL"],
        ["Serum Potassium", "4.2", "mEq/L", "3.5 - 5.0", "NORMAL"]
    ]
    p2_y = draw_table(page2, 85, "3. COMPREHENSIVE METABOLIC, RENAL & GLYCEMIC PANEL", cmp_headers, cmp_rows)
    
    # Clinical Interpretation
    page2.insert_text(fitz.Point(36, p2_y + 10), "CLINICAL INTERPRETATION & DIAGNOSTIC IMPRESSION", color=NAVY, fontsize=10.5, fontname="helv")
    
    box_rect = fitz.Rect(36, p2_y + 18, 576, p2_y + 160)
    page2.draw_rect(box_rect, color=BORDER_COLOR, fill=LIGHT_GRAY, width=1)
    
    interp_lines = [
        "DIAGNOSTIC SUMMARY:",
        "1. Severe Microcytic Hypochromic Anemia: Marked reduction in Hemoglobin (7.8 g/dL), Hematocrit (24.6%),",
        "   MCV (68.2 fL), and MCH (20.4 pg), consistent with red blood cell microcytosis and hypochromia.",
        "2. Profound Iron Deficiency: Critically depleted iron reserves evidenced by Serum Ferritin 4.8 ng/mL,",
        "   Serum Iron 18 ug/dL, elevated TIBC 485 ug/dL, and transferrin saturation 3.7%.",
        "3. Anisocytosis & Reactive Thrombocytosis: Elevated RDW (19.4%) and Platelet count (445 k/uL) reflect",
        "   active bone marrow response and erythropoietic stress secondary to chronic blood loss.",
        "4. Preserved Renal Function & Glycemic Control: Serum Creatinine (0.78 mg/dL), eGFR > 90, and HbA1c (5.3%)."
    ]
    text_y = p2_y + 34
    for il in interp_lines:
        is_bold_title = "DIAGNOSTIC SUMMARY:" in il
        page2.insert_text(fitz.Point(46, text_y), il, color=NAVY if is_bold_title else TEXT_DARK, fontsize=8.5, fontname="helv")
        text_y += 14
        
    # Recommendations
    page2.insert_text(fitz.Point(36, p2_y + 180), "RECOMMENDED CLINICAL ACTION PLAN & TESTS", color=NAVY, fontsize=10.5, fontname="helv")
    recs_rect = fitz.Rect(36, p2_y + 188, 576, p2_y + 275)
    page2.draw_rect(recs_rect, color=BORDER_COLOR, fill=(1, 1, 1), width=1)
    
    rec_lines = [
        "1. Therapeutic Iron Repletion: Initiate intravenous iron therapy (e.g., Ferric Carboxymaltose 1000 mg IV) given the severity",
        "   of anemia (Hb < 8.0 g/dL) and clinical intolerance or inadequacy of oral iron.",
        "2. Gynecological Evaluation: Immediate pelvic ultrasound and gynecological consultation to investigate the underlying etiology",
        "   of persistent menorrhagia.",
        "3. Gastrointestinal Workup: Stool Fecal Immunochemical Test (FIT) and GI consultation for endoscopy/colonoscopy to exclude",
        "   concomitant occult gastrointestinal hemorrhage.",
        "4. Follow-up: Repeat CBC and reticulocyte count at 2 weeks post-iron administration to verify marrow response."
    ]
    rec_y = p2_y + 204
    for rl in rec_lines:
        page2.insert_text(fitz.Point(46, rec_y), rl, color=TEXT_DARK, fontsize=8.5, fontname="helv")
        rec_y += 14
        
    # Electronic Signatures Box
    sig_rect = fitz.Rect(36, 680, 576, 735)
    page2.draw_rect(sig_rect, color=BORDER_COLOR, fill=LIGHT_GRAY, width=1)
    page2.insert_text(fitz.Point(46, 698), "Electronically Verified & Authenticated by:", color=TEXT_DARK, fontsize=8, fontname="helv")
    page2.insert_text(fitz.Point(46, 712), "Dr. Sarah Jenkins, MD, FCAP", color=NAVY, fontsize=9.5, fontname="helv")
    page2.insert_text(fitz.Point(46, 725), "Medical Director & Chief of Hematopathology | License #: MD-883492-NY", color=(0.3, 0.3, 0.3), fontsize=7.5, fontname="helv")
    
    page2.insert_text(fitz.Point(380, 712), "[DIGITALLY SIGNED & ENCRYPTED]", color=(0.1, 0.5, 0.2), fontsize=8.5, fontname="helv")
    page2.insert_text(fitz.Point(380, 725), "Verification Timestamp: 2026-10-26 14:05:12 UTC", color=(0.4, 0.4, 0.4), fontsize=7.5, fontname="helv")
    
    page2.insert_text(fitz.Point(36, 755), "Metropolis Healthcare System | Laboratory Medicine | Confidential Report", color=(0.5, 0.5, 0.5), fontsize=8, fontname="helv")
    page2.insert_text(fitz.Point(500, 755), "Page 2 of 2", color=(0.5, 0.5, 0.5), fontsize=8, fontname="helv")
    
    doc.save(output_path)
    doc.close()
    print(f"[+] Multi-page formal clinical PDF successfully generated: {output_path}")

if __name__ == "__main__":
    create_formal_clinical_pdf("Formal-Clinical-Patient-Report.pdf")
    create_formal_clinical_pdf("Test-MedicalPatient Report.pdf")
