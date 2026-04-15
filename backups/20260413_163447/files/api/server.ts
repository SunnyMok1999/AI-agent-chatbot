/**
 * local server entry file, for local development
 */
import app from './app.js';

/**
 * start server with port
 */
const PORT = process.env.PORT || 3001;

let server: ReturnType<typeof app.listen> | null = null;
let isShuttingDown = false;

const startServer = () => {
  server = app.listen(PORT, () => {
    console.log(`Server ready on port ${PORT}`);
  });

  server.on('error', (error: any) => {
    if (error?.code === 'EADDRINUSE') {
      console.warn(`Port ${PORT} is busy. Retrying bind in 800ms...`);
      setTimeout(() => {
        if (!isShuttingDown) {
          try {
            server?.close();
          } catch {
            // no-op
          }
          startServer();
        }
      }, 800);
      return;
    }

    throw error;
  });
};

startServer();

const gracefulShutdown = (signal: string, restart = false) => {
  if (isShuttingDown) return;
  isShuttingDown = true;
  console.log(`${signal} signal received`);

  if (!server) {
    if (restart) {
      process.kill(process.pid, 'SIGUSR2');
      return;
    }
    process.exit(0);
    return;
  }

  server.close(() => {
    console.log('Server closed');
    if (restart) {
      process.kill(process.pid, 'SIGUSR2');
    } else {
      process.exit(0);
    }
  });
};

/**
 * close server
 */
process.on('SIGTERM', () => {
  gracefulShutdown('SIGTERM');
});

process.on('SIGINT', () => {
  gracefulShutdown('SIGINT');
});

// Graceful restart support for nodemon
process.on('SIGUSR2', () => {
  gracefulShutdown('SIGUSR2', true);
});

export default app;