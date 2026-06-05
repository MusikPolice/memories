import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['tests/frontend/**/*.test.js'],
    coverage: {
      provider: 'v8',
      include: ['src/memories/frontend/chat.js'],
    },
  },
});
