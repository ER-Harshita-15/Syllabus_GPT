import { useState } from "react";
import Documentation from "./components/Documentation";
import SyllabusForm from "./components/SyllabusForm";
import RymdBackground from "./components/RymdBackground";

export default function App() {
  const [currentPage, setCurrentPage] = useState("documentation");

  return (
    <div className="min-h-screen relative overflow-hidden">
      
      {/* Animated background */}
      <RymdBackground />

      {/* Foreground content */}
      <main className="relative z-10">
        {currentPage === "documentation" && (
          <Documentation onNavigate={(page) => setCurrentPage(page)} />
        )}

        {currentPage === "syllabus-form" && (
          <SyllabusForm onNavigate={(page) => setCurrentPage(page)} />
        )}
      </main>
    </div>
  );
}
