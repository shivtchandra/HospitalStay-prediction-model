import React from "react";
import useRevealOnScroll from "../hooks/useRevealOnScroll";

export default function RevealSection({ children, className = "", delay = 0 }) {
  const [ref, visible] = useRevealOnScroll();

  return (
    <div
      ref={ref}
      className={`${className} transform transition-all duration-700 ease-out ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}
