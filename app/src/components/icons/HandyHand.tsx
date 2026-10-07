// Better Voice mark: a microphone with sound waves (used as the sidebar icon).
const HandyHand = ({
  width,
  height,
  className = "",
  colorClass = "fill-text stroke-text",
}: {
  width?: number | string;
  height?: number | string;
  className?: string;
  colorClass?: string;
}) => (
  <svg
    width={width || 126}
    height={height || 135}
    viewBox="0 0 126 135"
    className={`${colorClass} ${className}`}
    xmlns="http://www.w3.org/2000/svg"
  >
    <rect x="44" y="10" width="38" height="68" rx="19" />
    <path
      d="M28 62a35 35 0 0 0 70 0M63 97v22M45 121h36"
      fill="none"
      strokeWidth="8"
      strokeLinecap="round"
    />
    <path
      d="M106 44c7 11 7 31 0 42M14 44c-7 11-7 31 0 42"
      fill="none"
      strokeWidth="7"
      strokeLinecap="round"
      opacity="0.55"
    />
  </svg>
);

export default HandyHand;
