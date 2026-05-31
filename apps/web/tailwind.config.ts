import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        panel: "#f8f5ec",
        ink: "#111111",
        accent: "#e4ff4f",
        signal: "#15b392",
        alert: "#ff7d4d",
      },
      boxShadow: {
        panel: "10px 10px 0 #111111",
      },
      fontFamily: {
        display: ["Georgia", "serif"],
        body: ["'Trebuchet MS'", "Verdana", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
