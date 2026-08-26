/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: { 950: "#0b1829", 900: "#10243d", 800: "#173653", 700: "#244e6d" },
        teal: { 800: "#075b5d", 700: "#087477", 600: "#0b8585", 100: "#d8f0ed", 50: "#eff9f7" },
        sand: { 100: "#eee9df", 50: "#f8f6f1" },
      },
      boxShadow: { card: "0 1px 2px rgba(11,24,41,.04), 0 10px 30px rgba(11,24,41,.06)" },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
