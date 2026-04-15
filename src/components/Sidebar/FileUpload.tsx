import React, { useState } from 'react';
import { Upload, X, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface FileUploadProps {
  onClose: () => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onClose }) => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  const isSupportedFile = (candidate: File) => {
    const allowedMime = new Set([
      'application/pdf',
      'text/plain',
      'text/markdown',
      'image/png',
      'image/x-png',
    ]);

    if (allowedMime.has(candidate.type)) return true;
    const name = candidate.name.toLowerCase();
    return name.endsWith('.pdf') || name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.png');
  };

  const selectFile = (candidate: File) => {
    if (!isSupportedFile(candidate)) {
      setStatus('error');
      setMessage('Unsupported file format. Please upload PDF, TXT, MD, or PNG.');
      return;
    }

    setFile(candidate);
    setStatus('idle');
    setMessage('');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      selectFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const dropped = e.dataTransfer?.files?.[0];
    if (dropped) selectFile(dropped);
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus('uploading');
    setMessage('Processing document...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setStatus('success');
        setMessage('Document ingested successfully!');
        setTimeout(onClose, 2000);
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      setStatus('error');
      setMessage(error.message || 'Error processing document');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-md bg-white dark:bg-slate-900 rounded-2xl shadow-2xl p-6 relative overflow-hidden">
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
        >
          <X size={20} />
        </button>

        <h3 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">Upload Document</h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
          Upload PDF, TXT, MD, or PNG files to expand the chatbot's knowledge base.
        </p>

        <div 
          className={cn(
            "border-2 border-dashed rounded-xl p-8 transition-colors text-center cursor-pointer mb-6",
            isDragging
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30"
              : file
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-slate-300 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-600"
          )}
          onClick={() => document.getElementById('fileInput')?.click()}
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            id="fileInput"
            type="file"
            className="hidden"
            accept=".pdf,.txt,.md,.png"
            onChange={handleFileChange}
          />
          <div className="flex flex-col items-center gap-2">
            <Upload size={32} className={cn(file ? "text-blue-500" : "text-slate-400")} />
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              {file ? file.name : "Click to select or drag and drop"}
            </span>
            <span className="text-xs text-slate-400">PDF, TXT, MD, PNG (max 100MB)</span>
          </div>
        </div>

        {status !== 'idle' && (
          <div className={cn(
            "flex items-center gap-3 p-3 rounded-lg text-sm mb-6",
            status === 'uploading' && "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400",
            status === 'success' && "bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400",
            status === 'error' && "bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400"
          )}>
            {status === 'uploading' && <Loader2 size={18} className="animate-spin" />}
            {status === 'success' && <CheckCircle2 size={18} />}
            {status === 'error' && <AlertCircle size={18} />}
            {message}
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || status === 'uploading' || status === 'success'}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          {status === 'uploading' ? 'Uploading...' : 'Upload & Process'}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;
