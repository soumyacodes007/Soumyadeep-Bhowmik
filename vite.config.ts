// vite.config.ts
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), ''); // Use process.cwd() for current working directory
    return {
      define: {
        // Ensure GEMINI_API_KEY is correctly sourced from your .env file
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      // No need for resolve alias if your tsconfig paths handle it or not used
    };
});