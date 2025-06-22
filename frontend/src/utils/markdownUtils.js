// frontend/src/utils/markdownUtils.js

import React from 'react';

/**
 * Basit markdown-to-HTML converter
 * Gemini AI'dan gelen markdown formatını HTML'e çevirir
 */
export const parseMarkdownToHTML = (markdownText) => {
  if (!markdownText || typeof markdownText !== 'string') {
    return '';
  }

  let html = markdownText;

  // Headers
  html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');

  // Bold text
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Italic text
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Code blocks (triple backticks)
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

  // Inline code
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');

  // Unordered lists
  const lines = html.split('\n');
  const processedLines = [];
  let inList = false;
  let inOrderedList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Unordered list items
    if (line.match(/^[\s]*[-*+]\s+(.+)/)) {
      if (!inList) {
        processedLines.push('<ul>');
        inList = true;
      }
      const content = line.replace(/^[\s]*[-*+]\s+/, '');
      processedLines.push(`<li>${content}</li>`);
    }
    // Ordered list items
    else if (line.match(/^[\s]*\d+\.\s+(.+)/)) {
      if (!inOrderedList) {
        processedLines.push('<ol>');
        inOrderedList = true;
      }
      const content = line.replace(/^[\s]*\d+\.\s+/, '');
      processedLines.push(`<li>${content}</li>`);
    }
    // Regular lines
    else {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      if (inOrderedList) {
        processedLines.push('</ol>');
        inOrderedList = false;
      }
      
      // Empty lines become breaks, others become paragraphs
      if (line.trim() === '') {
        processedLines.push('<br/>');
      } else {
        processedLines.push(`<p>${line}</p>`);
      }
    }
  }

  // Close any open lists
  if (inList) {
    processedLines.push('</ul>');
  }
  if (inOrderedList) {
    processedLines.push('</ol>');
  }

  return processedLines.join('\n');
};

/**
 * React component olarak markdown render etmek için
 */
export const MarkdownRenderer = ({ content }) => {
  const htmlContent = parseMarkdownToHTML(content);
  
  return (
    <div 
      className="markdown-content"
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
};
