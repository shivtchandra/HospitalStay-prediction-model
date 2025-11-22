import React, { useRef, useState, useEffect } from "react";

export default function LazyImage({ src, alt = "", className = "", placeholderClass = "bg-neutral-800", ...props }) {
  const imgRef = useRef(null);
  const [visible, setVisible] = useState(false);
  const [loadedSrc, setLoadedSrc] = useState(null);

  useEffect(() => {
    if (!imgRef.current) return;
    const node = imgRef.current;

    // if IntersectionObserver isn't supported, just load immediately
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      setLoadedSrc(src);
      return;
    }

    const io = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setVisible(true);
          setLoadedSrc(src);
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "200px 0px" }); // preload slightly before visible

    io.observe(node);
    return () => io.disconnect();
  }, [src]);

  return (
    <div ref={imgRef} className={`w-full h-full ${placeholderClass} ${!visible ? "opacity-0" : "opacity-100"} transition-opacity duration-500`}>
      {loadedSrc ? (
        <img
          src={loadedSrc}
          alt={alt}
          className={className}
          loading="lazy"
          {...props}
        />
      ) : (
        // small inline skeleton to keep layout stable
        <div className="w-full h-full" aria-hidden="true" />
      )}
    </div>
  );
}
