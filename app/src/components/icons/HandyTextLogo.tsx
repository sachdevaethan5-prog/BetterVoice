import React from "react";

// The product name, drawn as part of the logo (not translated).
const APP_NAME = "Better Voice";

// Better Voice wordmark: a microphone with sound waves, then the name.
const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox="0 0 930 240"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Better Voice"
    >
      <rect
        x="62"
        y="20"
        width="76"
        height="130"
        rx="38"
        className="logo-primary"
      />
      <path
        d="M30 112a70 70 0 0 0 140 0M100 182v38M66 220h68"
        stroke="currentColor"
        strokeWidth="16"
        strokeLinecap="round"
      />
      <path
        d="M196 70c14 22 14 62 0 84M222 48c26 36 26 92 0 128"
        stroke="currentColor"
        strokeWidth="14"
        strokeLinecap="round"
        opacity="0.55"
      />
      <text
        x="290"
        y="150"
        fontFamily="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
        fontSize="112"
        fontWeight="700"
        fill="currentColor"
        letterSpacing="-2"
      >
        {APP_NAME}
      </text>
    </svg>
  );
};

export default HandyTextLogo;
