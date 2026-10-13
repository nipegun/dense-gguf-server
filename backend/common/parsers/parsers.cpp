#include "parsers.h"

#include "log.h"

void fForeachFunction(const json & pTools, const std::function<void(size_t, const json &)> & pFn) {
  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vTool = pTools[vI];
    if (!vTool.contains("type") || vTool.at("type") != "function" || !vTool.contains("function")) {
      LOG_INF("Skipping tool without function: %s", vTool.mDump(2).c_str());
      continue;
    }
    pFn(vI, vTool);
  }
}

void fForeachParameter(
  const json & pFunction,
  const std::function<void(size_t, const common_chat_schema_property &, const common_chat_schema_document_ptr &)> &
    pFn) {
  auto         vParams = fCommonChatToolParameters(pFunction);
  auto         vDoc    = std::make_shared<const common_chat_schema_document>(fCommonChatSchemaFromJson(vParams));
  const auto * vObject = dynamic_cast<const common_chat_schema_object *>(vDoc->vRoot.get());
  if (!vObject) {
    return;
  }
  for (size_t vI = 0; vI < vObject->lProperties.size(); vI++) {
    pFn(vI, vObject->lProperties[vI], vDoc);
  }
}
