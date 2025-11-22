import { useEffect, useRef, useState } from "react";

export default function useRevealOnScroll(rootMargin = "0px 0px -10% 0px") {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      return;
    }
    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin });

    io.observe(node);
    return () => io.disconnect();
  }, [ref, rootMargin]);

  return [ref, visible];
}
