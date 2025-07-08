const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Add secure APIs here if needed
}); 