/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1E40AF',
          foreground: '#FFFFFF',
        },
        muted: {
          DEFAULT: '#F1F5F9',
          foreground: '#64748B',
        },
      },
    },
  },
  plugins: [],
}
