import { useState } from "react";
import api from "../api/api";

export default function ExportButtons({ syllabus, subject, usePyq, topK }) {
  const [loading, setLoading] = useState(false);

  const exportPdf = async () => {
    setLoading(true);
    try {
      const res = await api.post(
        "/notes/generate-and-export/pdf",
        {
          syllabus_text: syllabus,
          subject: subject,
          use_pyq: usePyq,
          top_k: topK,
          filename: "study_notes.pdf",
        },
        { responseType: "blob" }
      );

      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "study_notes.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert("PDF export failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full rounded-2xl border border-white/20 bg-black/45 backdrop-blur-xl shadow-2xl overflow-hidden">
      
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/20">
        <h3 className="text-lg font-semibold text-white">
          Export Notes
        </h3>
        <p className="text-sm text-slate-300">
          Download your AI-generated notes as a professional PDF
        </p>
      </div>

      {/* Body */}
      <div className="px-6 py-6 space-y-4">
        <button
          onClick={exportPdf}
          disabled={loading || !syllabus.trim()}
          className="
              w-1/4 flex items-center justify-center gap-3
  px-7 py-4 rounded-xl
  bg-green-500/90
  text-white text-base font-semibold
  shadow-md shadow-indigo-500/30
  hover:bg-emerald-400
  hover:shadow-lg hover:shadow-indigo-500/40
  transition-all duration-200
  disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-amber-300/40 border-t-white rounded-full animate-spin" />
              <span>Exporting PDF…</span>
            </>
          ) : (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 16v-8m0 8-3-3m3 3 3-3M4 4h16v16H4z"
                />
              </svg>
              <span>Export PDF</span>
            </>
          )}
        </button>

        {/* Meta info */}
        <div className="text-xs text-slate-400 flex items-center justify-between">
          <span>High-quality formatting</span>
          <span>PDF · A4</span>
        </div>
      </div>
    </div>
  );
}
