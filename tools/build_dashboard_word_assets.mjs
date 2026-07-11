import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function colName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function rangeForMatrix(startCell, rowCount, colCount) {
  const match = /^([A-Z]+)(\d+)$/.exec(startCell);
  if (!match) {
    throw new Error(`Invalid startCell: ${startCell}`);
  }
  const [, startColLetters, startRowText] = match;
  let startCol = 0;
  for (const char of startColLetters) {
    startCol = startCol * 26 + (char.charCodeAt(0) - 64);
  }
  startCol -= 1;
  const startRow = Number(startRowText);
  const endCol = startCol + colCount - 1;
  const endRow = startRow + rowCount - 1;
  return `${startCell}:${colName(endCol)}${endRow}`;
}

function styleTitle(sheet, rangeText, text) {
  const range = sheet.getRange(rangeText);
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: "#0F172A",
    font: { name: "Aptos", size: 16, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
    rowHeightPx: 34,
  };
}

function styleSubtitle(sheet, rangeText, text) {
  const range = sheet.getRange(rangeText);
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: "#E2E8F0",
    font: { name: "Aptos", size: 10, color: "#334155" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
    rowHeightPx: 24,
  };
}

function styleSectionHeader(sheet, rangeText, text, fill = "#DCEBFF") {
  const range = sheet.getRange(rangeText);
  range.merge();
  range.values = [[text]];
  range.format = {
    fill,
    font: { name: "Aptos", size: 12, bold: true, color: "#0F172A" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
    rowHeightPx: 26,
  };
}

function bandedRows(block, headerFill = "#1D4ED8") {
  block.getRow(0).format = {
    fill: headerFill,
    font: { name: "Aptos", size: 11, bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
  };
  for (let rowIndex = 1; rowIndex < block.rowCount; rowIndex += 1) {
    block.getRow(rowIndex).format = {
      fill: rowIndex % 2 === 1 ? "#F8FAFC" : "#FFFFFF",
      font: { name: "Aptos", size: 10, color: "#0F172A" },
      verticalAlignment: "top",
      wrapText: true,
    };
  }
}

function setColumnWidths(sheet, specs, maxRow = 400) {
  for (const [column, widthPx] of specs) {
    sheet.getRange(`${column}1:${column}${maxRow}`).format.columnWidthPx = widthPx;
  }
}

function fillDiffColumn(block, columnIndex, dataStartRowIndex = 1) {
  for (let rowIndex = dataStartRowIndex; rowIndex < block.rowCount; rowIndex += 1) {
    const cell = block.getCell(rowIndex, columnIndex);
    const raw = cell.values?.[0]?.[0];
    const value = typeof raw === "number" ? raw : Number(raw);
    if (Number.isNaN(value)) {
      continue;
    }
    if (value > 0) {
      cell.format.fill = "#DCFCE7";
      cell.format.font = { name: "Aptos", size: 10, bold: true, color: "#166534" };
    } else if (value < 0) {
      cell.format.fill = "#FEE2E2";
      cell.format.font = { name: "Aptos", size: 10, bold: true, color: "#991B1B" };
    } else {
      cell.format.fill = "#F1F5F9";
      cell.format.font = { name: "Aptos", size: 10, color: "#475569" };
    }
  }
}

function addCountCard(sheet, rangeText, label, value, fill, valueColor = "#0F172A") {
  const range = sheet.getRange(rangeText);
  range.merge();
  range.values = [[`${label}\n${value}`]];
  range.format = {
    fill,
    font: { name: "Aptos", size: 13, bold: true, color: valueColor },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}

function statusPalette(status) {
  if (status === "degraded") {
    return { fill: "#FEE2E2", title: "Degraded", titleColor: "#991B1B" };
  }
  if (status === "improved") {
    return { fill: "#DCFCE7", title: "Improved", titleColor: "#166534" };
  }
  return { fill: "#E2E8F0", title: "Stable", titleColor: "#334155" };
}

function writeLabelValueRow(sheet, row, label, value) {
  const labelRange = sheet.getRange(`A${row}:B${row}`);
  const valueRange = sheet.getRange(`C${row}:J${row}`);
  labelRange.merge();
  valueRange.merge();
  labelRange.values = [[label]];
  valueRange.values = [[value]];
  labelRange.format = {
    fill: "#F8FAFC",
    font: { name: "Aptos", size: 10, bold: true, color: "#0F172A" },
    verticalAlignment: "top",
    wrapText: true,
  };
  valueRange.format = {
    fill: "#FFFFFF",
    font: { name: "Aptos", size: 10, color: "#0F172A" },
    verticalAlignment: "top",
    wrapText: true,
  };
}

function writeFigure4(sheet, figure) {
  styleTitle(sheet, "A1:K1", figure.title);
  styleSubtitle(sheet, "A2:K2", figure.subtitle);
  const matrix = [figure.headers, ...figure.rows];
  const block = sheet.getRange("A4").write(matrix);
  bandedRows(block);
  block.getColumn(0).format.horizontalAlignment = "left";
  for (let columnIndex = 1; columnIndex < figure.headers.length; columnIndex += 1) {
    block.getColumn(columnIndex).format.numberFormat = "0.000";
    block.getColumn(columnIndex).format.horizontalAlignment = "center";
  }
  setColumnWidths(
    sheet,
    [
      ["A", 420],
      ["B", 115],
      ["C", 115],
      ["D", 120],
      ["E", 120],
      ["F", 120],
      ["G", 120],
      ["H", 120],
      ["I", 120],
      ["J", 125],
      ["K", 125],
    ],
    40,
  );
  sheet.freezePanes.freezeRows(4);
}

function writeFigure5(sheet, figure, counts) {
  styleTitle(sheet, "A1:D1", figure.title);
  styleSubtitle(sheet, "A2:D2", figure.subtitle);
  addCountCard(sheet, "A4:B5", "Improved", String(counts.improved), "#DCFCE7", "#166534");
  addCountCard(sheet, "C4:D5", "Degraded", String(counts.degraded), "#FEE2E2", "#991B1B");
  const block = sheet.getRange("A7").write([figure.headers, ...figure.rows]);
  bandedRows(block, "#0F766E");
  block.getColumn(1).format.numberFormat = "0.000";
  block.getColumn(2).format.numberFormat = "0.000";
  block.getColumn(3).format.numberFormat = "+0.000;-0.000;0.000";
  fillDiffColumn(block, 3);
  setColumnWidths(
    sheet,
    [
      ["A", 220],
      ["B", 105],
      ["C", 105],
      ["D", 105],
    ],
    30,
  );
  sheet.freezePanes.freezeRows(7);
}

function writeFigure6(sheet, figure) {
  styleTitle(sheet, "A1:J1", figure.title);
  styleSubtitle(sheet, "A2:J2", figure.subtitle);
  addCountCard(sheet, "A4:C6", "Improved", String(figure.counts.improved), "#DCFCE7", "#166534");
  addCountCard(sheet, "D4:F6", "Degraded", String(figure.counts.degraded), "#FEE2E2", "#991B1B");
  addCountCard(sheet, "G4:I6", "Stable", String(figure.counts.stable), "#E2E8F0", "#334155");

  setColumnWidths(
    sheet,
    [
      ["A", 120],
      ["B", 120],
      ["C", 110],
      ["D", 110],
      ["E", 120],
      ["F", 120],
      ["G", 130],
      ["H", 130],
      ["I", 130],
      ["J", 130],
    ],
    220,
  );

  let row = 8;
  for (const card of figure.cards) {
    const palette = statusPalette(card.status);
    const titleRange = sheet.getRange(`A${row}:J${row}`);
    titleRange.merge();
    titleRange.values = [[`${palette.title} | ${card.question}`]];
    titleRange.format = {
      fill: palette.fill,
      font: { name: "Aptos", size: 12, bold: true, color: palette.titleColor },
      horizontalAlignment: "left",
      verticalAlignment: "center",
      wrapText: true,
      rowHeightPx: 28,
    };

    writeLabelValueRow(sheet, row + 1, "Status", palette.title);
    writeLabelValueRow(
      sheet,
      row + 2,
      "Relevant doc IDs",
      card.relevant_doc_ids.join(", ") || "None",
    );
    writeLabelValueRow(
      sheet,
      row + 3,
      "Retrieved doc IDs (V1)",
      card.baseline_retrieved_doc_ids.join(", ") || "None",
    );
    writeLabelValueRow(
      sheet,
      row + 4,
      "Retrieved doc IDs (V2)",
      card.updated_retrieved_doc_ids.join(", ") || "None",
    );
    writeLabelValueRow(
      sheet,
      row + 5,
      "Degraded metrics",
      card.degraded_metrics.join(", ") || "None",
    );

    const metricBlock = sheet.getRange(`A${row + 7}`).write([
      ["Metric", "V1", "V2", "Diff"],
      ...card.metrics.map((metric) => [
        metric.metric,
        metric.v1,
        metric.v2,
        metric.diff,
      ]),
    ]);
    bandedRows(metricBlock, "#475569");
    metricBlock.getColumn(1).format.numberFormat = "0.000";
    metricBlock.getColumn(2).format.numberFormat = "0.000";
    metricBlock.getColumn(3).format.numberFormat = "+0.000;-0.000;0.000";
    fillDiffColumn(metricBlock, 3);

    const answerLabel1 = sheet.getRange(`E${row + 7}:F${row + 7}`);
    const answerValue1 = sheet.getRange(`G${row + 7}:J${row + 9}`);
    answerLabel1.merge();
    answerValue1.merge();
    answerLabel1.values = [["V1 Answer"]];
    answerValue1.values = [[card.baseline_answer]];
    answerLabel1.format = {
      fill: "#DBEAFE",
      font: { name: "Aptos", size: 10, bold: true, color: "#1D4ED8" },
      verticalAlignment: "top",
      wrapText: true,
    };
    answerValue1.format = {
      fill: "#FFFFFF",
      font: { name: "Aptos", size: 10, color: "#0F172A" },
      verticalAlignment: "top",
      wrapText: true,
    };

    const answerLabel2 = sheet.getRange(`E${row + 10}:F${row + 10}`);
    const answerValue2 = sheet.getRange(`G${row + 10}:J${row + 12}`);
    answerLabel2.merge();
    answerValue2.merge();
    answerLabel2.values = [["V2 Answer"]];
    answerValue2.values = [[card.updated_answer]];
    answerLabel2.format = {
      fill: "#FFEDD5",
      font: { name: "Aptos", size: 10, bold: true, color: "#C2410C" },
      verticalAlignment: "top",
      wrapText: true,
    };
    answerValue2.format = {
      fill: "#FFFFFF",
      font: { name: "Aptos", size: 10, color: "#0F172A" },
      verticalAlignment: "top",
      wrapText: true,
    };

    row += 15;
  }
}

function writeTableWithHeader(sheet, startCell, title, headers, rows, widths, maxRow) {
  const titleMatch = /^([A-Z]+)(\d+)$/.exec(startCell);
  if (!titleMatch) {
    throw new Error(`Invalid startCell: ${startCell}`);
  }
  const startRow = Number(titleMatch[2]);
  const endCol = colName(headers.length - 1);
  styleSectionHeader(sheet, `${titleMatch[1]}${startRow}:${endCol}${startRow}`, title, "#E0F2FE");
  const block = sheet.getRange(`${titleMatch[1]}${startRow + 1}`).write([headers, ...rows]);
  bandedRows(block, "#0369A1");
  setColumnWidths(sheet, widths, maxRow);
  return block;
}

function writeFigure7(sheet, figure) {
  styleTitle(sheet, "A1:H1", figure.title);
  styleSubtitle(sheet, "A2:H2", figure.subtitle);
  const summary = sheet.getRange("A3:H4");
  summary.merge();
  summary.values = [[figure.summary_text]];
  summary.format = {
    fill: "#F8FAFC",
    font: { name: "Aptos", size: 11, color: "#0F172A" },
    verticalAlignment: "center",
    wrapText: true,
  };

  addCountCard(sheet, "A6:B7", "Added", String(figure.counts.added), "#DCFCE7", "#166534");
  addCountCard(sheet, "C6:D7", "Modified", String(figure.counts.modified), "#FEF3C7", "#92400E");
  addCountCard(sheet, "E6:F7", "Removed", String(figure.counts.removed), "#FEE2E2", "#991B1B");
  addCountCard(sheet, "G6:H7", "Unchanged", String(figure.counts.unchanged), "#E2E8F0", "#334155");

  const widthsAdded = [
    ["A", 120],
    ["B", 260],
    ["C", 520],
  ];
  const widthsModified = [
    ["A", 120],
    ["B", 240],
    ["C", 240],
    ["D", 160],
  ];

  let row = 9;
  writeTableWithHeader(
    sheet,
    `A${row}`,
    "Added Documents",
    figure.added.headers,
    figure.added.rows,
    widthsAdded,
    60,
  );
  row += figure.added.rows.length + 4;
  writeTableWithHeader(
    sheet,
    `A${row}`,
    "Modified Documents",
    figure.modified.headers,
    figure.modified.rows,
    widthsModified,
    120,
  );
  row += figure.modified.rows.length + 4;
  writeTableWithHeader(
    sheet,
    `A${row}`,
    "Removed Documents",
    figure.removed.headers,
    figure.removed.rows,
    widthsAdded,
    180,
  );
}

function writeCard(sheet, titleRange, valueRange, detailRange, label, value, detail, fill, valueColor) {
  const titleCell = sheet.getRange(titleRange);
  const valueCell = sheet.getRange(valueRange);
  const detailCell = sheet.getRange(detailRange);
  titleCell.merge();
  valueCell.merge();
  detailCell.merge();
  titleCell.values = [[label]];
  valueCell.values = [[value]];
  detailCell.values = [[detail]];
  titleCell.format = {
    fill,
    font: { name: "Aptos", size: 11, bold: true, color: valueColor },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  valueCell.format = {
    fill,
    font: { name: "Aptos", size: 18, bold: true, color: valueColor },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  detailCell.format = {
    fill,
    font: { name: "Aptos", size: 10, color: valueColor },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}

function writeFigure10(sheet, figure) {
  styleTitle(sheet, "A1:K1", figure.title);
  styleSubtitle(sheet, "A2:K2", figure.subtitle);

  writeCard(
    sheet,
    "A4:C4",
    "A5:C6",
    "A7:C8",
    figure.cards[0].label,
    figure.cards[0].value,
    figure.cards[0].detail,
    "#DBEAFE",
    "#1D4ED8",
  );
  writeCard(
    sheet,
    "E4:G4",
    "E5:G6",
    "E7:G8",
    figure.cards[1].label,
    figure.cards[1].value,
    figure.cards[1].detail,
    "#FEF3C7",
    "#92400E",
  );
  writeCard(
    sheet,
    "I4:K4",
    "I5:K6",
    "I7:K8",
    figure.cards[2].label,
    figure.cards[2].value,
    figure.cards[2].detail,
    "#DCFCE7",
    "#166534",
  );

  const categoryBlock = writeTableWithHeader(
    sheet,
    "A11",
    "Category Summary",
    ["Category", "Vectors"],
    figure.category_summary,
    [
      ["A", 140],
      ["B", 110],
    ],
    40,
  );
  categoryBlock.getColumn(1).format.numberFormat = "#,##0";

  const trackedBlock = writeTableWithHeader(
    sheet,
    "D11",
    "Tracked Collections",
    ["Collection", "Vectors"],
    figure.tracked_collections,
    [
      ["D", 300],
      ["E", 110],
    ],
    80,
  );
  trackedBlock.getColumn(1).format.numberFormat = "#,##0";

  addCountCard(
    sheet,
    "G11:I14",
    "Historical Result Rows",
    `${figure.historical_result_rows}`,
    "#EDE9FE",
    "#5B21B6",
  );

  setColumnWidths(
    sheet,
    [
      ["A", 120],
      ["B", 110],
      ["C", 110],
      ["D", 300],
      ["E", 110],
      ["F", 20],
      ["G", 120],
      ["H", 120],
      ["I", 120],
      ["J", 120],
      ["K", 120],
    ],
    90,
  );
}

function writeContents(sheet, payload) {
  styleTitle(sheet, "A1:F1", "Dashboard Figure Exports for Word");
  styleSubtitle(
    sheet,
    "A2:F2",
    "Workbook sheets correspond to the dashboard figures that did not have direct image exports.",
  );
  const matrix = [
    ["Figure", "Sheet", "Format", "Primary Source", "Comparison Timestamp", "Regression Timestamp"],
    [
      "Figure 4",
      "Figure4_DetailedScores",
      "Table",
      payload.metadata.comparison_sources.join(" + "),
      payload.metadata.comparison_timestamp,
      payload.metadata.regression_timestamp,
    ],
    [
      "Figure 5",
      "Figure5_OverallChanges",
      "Table",
      payload.metadata.regression_source,
      payload.metadata.comparison_timestamp,
      payload.metadata.regression_timestamp,
    ],
    [
      "Figure 6",
      "Figure6_StatusCards",
      "Card layout",
      payload.metadata.regression_source,
      payload.metadata.comparison_timestamp,
      payload.metadata.regression_timestamp,
    ],
    [
      "Figure 7",
      "Figure7_ChangeAnalysis",
      "Analysis tables",
      payload.metadata.regression_source,
      payload.metadata.comparison_timestamp,
      payload.metadata.regression_timestamp,
    ],
    [
      "Figure 10",
      "Figure10_ScaleCards",
      "Card layout",
      "Live scale dashboard backing data",
      payload.metadata.comparison_timestamp,
      payload.metadata.regression_timestamp,
    ],
  ];
  const block = sheet.getRange("A4").write(matrix);
  bandedRows(block, "#334155");
  setColumnWidths(
    sheet,
    [
      ["A", 100],
      ["B", 210],
      ["C", 120],
      ["D", 320],
      ["E", 140],
      ["F", 140],
    ],
    20,
  );
}

async function main() {
  if (process.argv.length < 4) {
    throw new Error("Usage: build_dashboard_word_assets.mjs <input_json> <output_xlsx>");
  }
  const inputJsonPath = path.resolve(process.argv[2]);
  const outputXlsxPath = path.resolve(process.argv[3]);
  const outputDir = path.dirname(outputXlsxPath);
  await fs.mkdir(outputDir, { recursive: true });

  const payload = JSON.parse(await fs.readFile(inputJsonPath, "utf8"));
  const workbook = Workbook.create();

  const contents = workbook.worksheets.add("Contents");
  const figure4 = workbook.worksheets.add("Figure4_DetailedScores");
  const figure5 = workbook.worksheets.add("Figure5_OverallChanges");
  const figure6 = workbook.worksheets.add("Figure6_StatusCards");
  const figure7 = workbook.worksheets.add("Figure7_ChangeAnalysis");
  const figure10 = workbook.worksheets.add("Figure10_ScaleCards");

  writeContents(contents, payload);
  writeFigure4(figure4, payload.figure4);
  writeFigure5(figure5, payload.figure5, payload.figure6.counts);
  writeFigure6(figure6, payload.figure6);
  writeFigure7(figure7, payload.figure7);
  writeFigure10(figure10, payload.figure10);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputXlsxPath);

  const summary = await workbook.inspect({
    kind: "sheet,region",
    maxChars: 6000,
    tableMaxRows: 6,
    tableMaxCols: 6,
    tableMaxCellChars: 80,
  });
  await fs.writeFile(
    path.join(outputDir, "workbook_inspect.txt"),
    typeof summary === "string" ? summary : JSON.stringify(summary, null, 2),
    "utf8",
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
