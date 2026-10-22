#pragma once

// Thin accessors over DuckDB APIs (v2.0+). Most wrap member accessors or Identifier conversions so call sites
// stay uniform; they are the single place to touch when the DuckDB API moves again.

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/types/size.hpp"
#include "duckdb/common/identifier.hpp"
#include "duckdb/common/optional.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/catalog/catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parser.hpp"

#define MONGO_FUNCTION_NAME(func)    ((func).GetName().GetIdentifierName())
#define MONGO_EXPR_RETURN_TYPE(expr) ((expr).GetReturnType())
#define MONGO_EXPR_TYPE(expr)        ((expr).GetExpressionType())

namespace duckdb {

// --- Vector helpers ---

inline bool MongoVectorHasAuxiliary(Vector &vec) {
	return vec.GetBufferRef().get() != nullptr;
}

// Initialize a vector without zeroing memory.
inline void MongoVectorInitializeUninitialized(Vector &vec, idx_t capacity = STANDARD_VECTOR_SIZE) {
	vec.Initialize(VectorDataInitialization::UNINITIALIZED, capacity);
}

template <typename T>
inline T *MongoFlatVectorGetDataMutable(Vector &vec) {
	return FlatVector::GetDataMutable<T>(vec);
}

inline void MongoVectorReferenceSingleValue(Vector &vec, const Value &val) {
	vec.Reference(val, count_t(1));
}

// Set a flat vector's tracked size (DataChunk::Verify checks per-vector sizes).
inline void MongoSetVectorSize(Vector &vec, idx_t size) {
	FlatVector::SetSize(vec, size);
}

// --- Table function helpers ---

// Scan filters are always ExpressionFilter; TableFilter has no Copy().
inline unique_ptr<TableFilter> MongoCopyFilter(const TableFilter &filter) {
	auto &ef = static_cast<const ExpressionFilter &>(filter);
	return ef.Copy();
}

// Declare a table function's optional named parameters as typed kwargs.
inline void MongoAddNamedParameters(TableFunction &function,
                                    const vector<pair<const char *, LogicalType>> &parameters) {
	function.GetSignature().WithTypedKwargs("options", [&](TypedKwargs &options) {
		for (auto &parameter : parameters) {
			options.Add(parameter.first, parameter.second);
		}
	});
}

inline unique_ptr<CreateViewInfo> MongoViewInfoFromSelect(ClientContext &context, unique_ptr<CreateViewInfo> info) {
	Parser parser(context);
	return CreateViewInfo::FromSelect(parser, std::move(info));
}

// --- Expression helpers ---

// Comparisons and casts are BoundFunctionExpression; the Bound*Expression classes expose static helpers.
inline bool MongoIsComparisonExpr(const Expression &expr) {
	return BoundComparisonExpression::IsComparison(expr);
}

inline const Expression &MongoComparisonLeft(const Expression &expr) {
	return BoundComparisonExpression::Left(expr.Cast<BoundFunctionExpression>());
}

inline const Expression &MongoComparisonRight(const Expression &expr) {
	return BoundComparisonExpression::Right(expr.Cast<BoundFunctionExpression>());
}

inline bool MongoIsCastExpr(const Expression &expr) {
	return BoundCastExpression::IsCast(expr);
}

// Cast always has a non-null child; return a pointer for uniform call sites.
inline const Expression *MongoCastChild(const Expression &expr) {
	return &BoundCastExpression::Child(expr.Cast<BoundFunctionExpression>());
}

inline string MongoExprName(const Expression &expr) {
	return expr.GetName().GetIdentifierName();
}

inline const string &MongoStructChildName(const LogicalType &type, idx_t index) {
	return StructType::GetChildName(type, index).GetIdentifierName();
}

inline string MongoCatalogEntryName(const CatalogEntry &entry) {
	return entry.name.GetIdentifierName();
}

inline const Value &MongoConstantValue(const BoundConstantExpression &expr) {
	return expr.GetValue();
}

inline const vector<unique_ptr<Expression>> &MongoConjunctionChildren(const BoundConjunctionExpression &expr) {
	return expr.GetChildren();
}

inline const ColumnBinding &MongoColumnBinding(const BoundColumnRefExpression &expr) {
	return expr.Binding();
}

inline ColumnBinding &MongoColumnBindingMutable(BoundColumnRefExpression &expr) {
	return expr.BindingMutable();
}

inline const unique_ptr<Expression> &MongoAggregateFilter(const BoundAggregateExpression &expr) {
	return expr.GetFilter();
}

inline const unique_ptr<BoundOrderModifier> &MongoAggregateOrderBys(const BoundAggregateExpression &expr) {
	return expr.GetOrderBys();
}

inline const vector<unique_ptr<Expression>> &MongoAggregateChildren(const BoundAggregateExpression &expr) {
	return expr.GetChildren();
}

inline decltype(auto) MongoAggregateFunction(const BoundAggregateExpression &expr) {
	return (expr.Function());
}

inline const vector<unique_ptr<Expression>> &MongoFuncChildren(const BoundFunctionExpression &expr) {
	return expr.GetChildren();
}

inline decltype(auto) MongoFuncFunction(const BoundFunctionExpression &expr) {
	return (expr.Function());
}

// --- Identifier helpers ---

using MongoColumnNameVector = vector<Identifier>;
inline MongoColumnNameVector MongoMakeColumnNames(const vector<string> &names) {
	MongoColumnNameVector result;
	result.reserve(names.size());
	for (auto &n : names) {
		result.emplace_back(Identifier(n));
	}
	return result;
}

template <typename T>
inline void MongoChildListAppend(child_list_t<T> &list, const string &name, T type) {
	list.push_back({Identifier(name), std::move(type)});
}
inline const string &MongoChildPairName(const pair<Identifier, LogicalType> &p) {
	return p.first.GetIdentifierName();
}

// --- Catalog info helpers ---

inline void MongoSetViewSchema(CreateViewInfo &info, const string &schema) {
	info.SetSchema(Identifier(schema));
}
inline void MongoSetViewName(CreateViewInfo &info, const string &name) {
	info.SetName(Identifier(name));
}
inline string MongoGetSchemaName(const CreateSchemaInfo &info) {
	return info.SchemaName().GetIdentifierName();
}
inline void MongoSetSchemaName(CreateSchemaInfo &info, const string &schema) {
	info.SetSchema(Identifier(schema));
}
inline string MongoGetDropName(const DropInfo &info) {
	return info.GetQualifiedName().Name().GetIdentifierName();
}
inline string MongoGetViewSchema(const CreateViewInfo &info) {
	return info.GetQualifiedName().Schema().GetIdentifierName();
}
inline string MongoGetViewName(const CreateViewInfo &info) {
	return info.GetViewName().GetIdentifierName();
}
using MongoDefaultEntryList = vector<Identifier>;
inline MongoDefaultEntryList MongoMakeDefaultEntries(const vector<string> &names) {
	MongoDefaultEntryList result;
	result.reserve(names.size());
	for (auto &n : names) {
		result.emplace_back(Identifier(n));
	}
	return result;
}

// Value::DefaultTryCastAs returns optional<Value>; keep the bool + out-param shape for call sites.
inline bool MongoDefaultTryCastAs(const Value &value, const LogicalType &target_type, Value &result,
                                  string *error_message = nullptr, bool strict = false) {
	auto casted = value.DefaultTryCastAs(target_type, error_message, strict);
	if (casted) {
		result = std::move(*casted);
		return true;
	}
	return false;
}

} // namespace duckdb
