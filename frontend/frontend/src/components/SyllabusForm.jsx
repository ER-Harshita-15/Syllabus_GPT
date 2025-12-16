import { useState } from "react";
import api from "../api/api";
import NotesViewer from "./NoteViewer";
import ExportButtons from "./ExportButtons";

export default function SyllabusForm({ onNavigate }) {
  const [syllabus, setSyllabus] = useState("");
  const [subject, setSubject] = useState("ML");
  const [usePyq, setUsePyq] = useState(true);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [textareaHeight, setTextareaHeight] = useState("auto");

  const handleSyllabusChange = (e) => {
    const value = e.target.value;
    setSyllabus(value);

    const textarea = e.target;
    const lineHeight = 20;
    const maxLines = 2;
    const maxHeight = lineHeight * maxLines;

    textarea.style.height = "auto";
    const scrollHeight = textarea.scrollHeight;
    const newHeight = Math.min(scrollHeight, maxHeight);

    setTextareaHeight(`${newHeight}px`);
    textarea.style.height = `${newHeight}px`;
  };

  const generateNotes = async () => {
    setLoading(true);
    try {
      const res = await api.post("/notes/generate", {
        syllabus_text: syllabus,
        subject,
        use_pyq: usePyq,
        top_k: 12,
      });
      setNotes(res.data.notes_markdown);

      setTimeout(() => {
        const footer = document.getElementById("footer");
        if (footer) footer.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch {
      alert("Error generating notes");
    }
    setLoading(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12 relative text-white">

      {/* ================= HEADER ================= */}
      <div className="flex items-center justify-between mb-10">
        <div>
          <h2 className="text-3xl font-semibold">Syllabus GPT</h2>
          <p className="text-slate-300 text-sm">
            Generate structured, exam-ready notes using AI
          </p>
        </div>

        <button
          onClick={() => onNavigate("documentation")}
          className="px-4 py-2 rounded-lg border border-white/20 text-white/80 hover:bg-white/5 transition"
        >
          ← Back
        </button>
      </div>

      {/* ================= NOTES OUTPUT ================= */}
      <div className="space-y-8">
        {notes && (
          <>
            <NotesViewer notes={notes} />
            <ExportButtons
              syllabus={syllabus}
              subject={subject}
              usePyq={usePyq}
              topK={12}
            />
          </>
        )}
      </div>

      {/* ================= FOOTER ================= */}
      <footer
        id="footer"
        className="mt-30 rounded-2xl border border-transparent bg-transparent p-8 text-center"
      >
       
      </footer>

      {/* ================= FIXED INPUT BAR ================= */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 w-[95%] max-w-7xl z-50">
        <div className="flex gap-4 bg-black/40 backdrop-blur-xl border border-white/15 rounded-2xl p-4">

          {/* LEFT CONTROLS */}
          <div className="w-64 space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Subject
              </label>
              <select
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full rounded-xl bg-white/10 border border-white/20 p-3 text-sm focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ML">Machine Learning</option>
                <option value="AI">Artificial Intelligence</option>
                <option value="IOT">IoT</option>
                <option value="TOC">TOC</option>
                <option value="STDS">Statistics</option>
              </select>
            </div>

            <label className="flex items-center gap-3 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={usePyq}
                onChange={() => setUsePyq(!usePyq)}
                className="w-4 h-4 rounded border-white/30 bg-white/10 checked:bg-indigo-500"
              />
              Include PYQs
            </label>
          </div>

          {/* TEXTAREA + ACTION */}
          <div className="flex-1 space-y-3">
            <textarea
              className="w-full rounded-2xl bg-white/10 border border-white/20 p-4 resize-none text-sm leading-relaxed placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500"
              style={{ minHeight: "60px", height: textareaHeight }}
              placeholder="Paste your syllabus here… (Unit-wise topics work best)"
              value={syllabus}
              onChange={handleSyllabusChange}
            />

            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">
                {syllabus.length} characters
              </span>

              <button
                onClick={generateNotes}
                disabled={loading || !syllabus.trim()}
                className=" w-1/4 h-0.8 flex items-center justify-center gap-3
  px-7 py-4 rounded-xl
  bg-green-500/90
  text-white text-base font-semibold
  shadow-md shadow-indigo-500/30
  hover:bg-emerald-400
  hover:shadow-lg hover:shadow-indigo-500/40
  transition-all duration-200
  disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Generating…" : "Generate Notes"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
