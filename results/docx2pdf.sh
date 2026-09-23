#!/bin/bash
# Word (AppleScript) で docx → PDF 変換し、ページ数を表示する
# 使い方: results/docx2pdf.sh results/thesis.docx
in="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
out="${in%.docx}.pdf"
osascript <<OSA
tell application "Microsoft Word"
  open POSIX file "$in"
  set d to active document
  save as d file name "$out" file format format PDF
  close d saving no
end tell
OSA
pdfinfo "$out" | grep Pages
