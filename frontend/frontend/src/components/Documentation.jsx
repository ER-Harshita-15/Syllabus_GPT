import { useState } from "react";

export default function Documentation({ onNavigate }) {
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="max-w-7xl mx-auto px-6 py-16 text-white">
      
      {/* ================= HERO ================= */}
      <section className="text-center max-w-3xl mx-auto mb-24">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-xl bg-transparent hover:bg-blue-950 border border-indigo-400/30 mb-6">
          <svg
  xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 24 24"
  className="w-8 h-8 text-amber-400"
  fill="currentColor"
>
  <path d="M11.26 6.86 7.87 4.86a.75.75 0 1 1 .78-1.28l3.36 2z" />
  <path d="M11.52 10.31 4.69 6.36a.75.75 0 0 1 .75-1.3l6.08 3.52z" />
  <path d="M19.6 5.33a.76.76 0 0 0-1-.27L12.75 8.4V6.85l3.39-2a.74.74 0 0 0 .25-1 .75.75 0 0 0-1-.25L11.65 5.78a.79.79 0 0 0-.39.68V21a.75.75 0 1 0 1.49 0V10.14l6.57-3.78a.76.76 0 0 0 .28-1.03z" />
  <circle cx="12.01" cy="3.09" r="0.84" />
  <path d="M9.48 10.82a.73.73 0 0 1 .38.64v4.08a.74.74 0 0 1-.38.65.79.79 0 0 1-.37.1.86.86 0 0 1-.38-.1L4.41 13.7a.75.75 0 1 1 .75-1.3l3.2 1.84V11.9l-1.45-.83a.75.75 0 0 1-.27-1 .74.74 0 0 1 1-.27z" />
  <circle cx="5.26" cy="9.26" r="0.84" />
  <path d="M16.34 9.77a.74.74 0 0 1 1 .27.75.75 0 0 1-.27 1l-1.45.83v2.34l3.2-1.84a.75.75 0 1 1 .75 1.3l-4.32 2.49a.86.86 0 0 1-.38.1.79.79 0 0 1-.37-.1.74.74 0 0 1-.38-.65V11.46a.73.73 0 0 1 .38-.64z" />
  <circle cx="18.74" cy="9.26" r="0.84" />
</svg>
        </div>

        <h1 className="text-5xl font-bold tracking-tight mb-4">
          Syllabus GPT
        </h1>

        <p className="text-lg text-slate-300 leading-relaxed">
          AI-powered platform that converts your syllabus into
          structured, exam-ready notes — fast, clear, and focused.
        </p>

        <div className="mt-10 flex justify-center gap-4">
          <button
            onClick={() => onNavigate("syllabus-form")}
            className="px-8 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold transition"
          >
            Generate Notes
          </button>

          <button
            onClick={() => window.scrollTo({ top: 600, behavior: "smooth" })}
            className="px-8 py-3 rounded-xl border border-white/20 text-white/80 hover:bg-white/5 transition"
          >
            Learn More
          </button>
        </div>
      </section>

      {/* ================= FEATURES ================= */}
      <section className="grid md:grid-cols-3 gap-6 mb-24">
        {[
          {
            title: "Fast & Exam-Focused",
            text: "Get concise, structured notes optimized for revision and exams.",
            icon: "⚡",
          },
          {
            title: "Subject Aware AI",
            text: "Supports ML, AI, IoT, TOC, Statistics with relevant explanations.",
            icon: "📖",
          },
          {
            title: "PYQ Integration",
            text: "Generate notes aligned with previous year exam patterns.",
            icon: "🧠",
          },
        ].map((item, i) => (
          <div
            key={i}
            className="rounded-2xl border border-white/15 bg-white/5 p-6 hover:bg-white/10 transition"
          >
            <div className="text-2xl mb-4">{item.icon}</div>
            <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              {item.text}
            </p>
          </div>
        ))}
      </section>

      {/* ================= HOW IT WORKS ================= */}
      <section className="grid lg:grid-cols-2 gap-12 mb-24">
        
        <div>
          <h2 className="text-2xl font-semibold mb-6">
            How it works
          </h2>

          <div className="space-y-4">
            {[
              "Paste your syllabus content",
              "Select subject and preferences",
              "Generate structured notes using AI",
              "Export and revise as PDF",
            ].map((step, i) => (
              <div
                key={i}
                className="flex items-center gap-4 p-4 rounded-xl border border-white/15 bg-white/5"
              >
                <div className="w-8 h-8 flex items-center justify-center rounded-full bg-indigo-600 text-sm font-bold">
                  {i + 1}
                </div>
                <span className="text-slate-200">{step}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ================= QUICK START ================= */}
        <div className="rounded-2xl border border-white/15 bg-white/5 p-8">
          <h3 className="text-xl font-semibold mb-4">
            Quick Start
          </h3>

          <p className="text-slate-300 mb-6">
            Start generating high-quality notes in minutes using a
            simple guided workflow.
          </p>

          <button
            onClick={() => onNavigate("syllabus-form")}
            className="px-8 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-semibold transition"
          >
            Start Generating Notes
          </button>
        </div>
      </section>

      {/* ================= TIPS ================= */}
      <section className="grid md:grid-cols-3 gap-6 mb-24">
        {[
          {
            title: "Clear Syllabus Input",
            text: "Mention units, topics, and subtopics for best output.",
            icon: "📝",
          },
          {
            title: "Correct Subject Choice",
            text: "Improves examples and conceptual depth.",
            icon: "🎯",
          },
          {
            title: "Use PYQs",
            text: "Boost exam relevance with past questions.",
            icon: "📚",
          },
        ].map((tip, i) => (
          <div
            key={i}
            className="rounded-2xl border border-white/15 bg-white/5 p-6"
          >
            <div className="text-xl mb-3">{tip.icon}</div>
            <h4 className="font-semibold mb-1">{tip.title}</h4>
            <p className="text-sm text-slate-300">{tip.text}</p>
          </div>
        ))}
      </section>

      {/* ================= FOOTER CTA ================= */}
      <section className="text-center">
        <p className="text-slate-300 mb-6">
          Built for students who prefer clarity over clutter.
        </p>

        <div className="flex justify-center gap-4">
          <button className="px-6 py-3 rounded-xl border border-white/20 text-white/80 hover:bg-white/5 transition">
            Documentation
          </button>
          <button className="px-6 py-3 rounded-xl border border-white/20 text-white/80 hover:bg-white/5 transition">
            Support
          </button>
        </div>
      </section>

    </div>
  );
}
