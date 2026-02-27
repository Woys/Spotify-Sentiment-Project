#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <tuple>

namespace py = pybind11;

std::vector<std::tuple<int, std::string, std::string>> scan_chunks(
    const std::vector<std::string>& texts,
    const std::unordered_map<std::string, std::vector<std::string>>& topic_words) {
    
    std::unordered_map<std::string, std::vector<std::string>> keyword_to_topics;
    size_t max_ngram = 1;
    
    for (const auto& pair : topic_words) {
        for (const auto& word : pair.second) {
            keyword_to_topics[word].push_back(pair.first);
            size_t spaces = 0;
            for (char c : word) if (c == ' ') spaces++;
            if (spaces + 1 > max_ngram) max_ngram = spaces + 1;
        }
    }

    std::vector<std::tuple<int, std::string, std::string>> results;
    py::gil_scoped_release release; 

    for (size_t i = 0; i < texts.size(); ++i) {
        const std::string& text = texts[i];
        if (text.empty()) continue;

        std::vector<std::string_view> tokens;
        size_t start = 0;
        bool in_word = false;

        for (size_t c = 0; c <= text.size(); ++c) {
            if (c < text.size() && std::isalnum(static_cast<unsigned char>(text[c]))) {
                if (!in_word) { start = c; in_word = true; }
            } else {
                if (in_word) {
                    tokens.emplace_back(text.data() + start, c - start);
                    in_word = false;
                }
            }
        }

        for (size_t j = 0; j < tokens.size(); ++j) {
            std::string ngram(tokens[j]);
            
            auto it = keyword_to_topics.find(ngram);
            if (it != keyword_to_topics.end()) {
                for (const auto& t : it->second) results.emplace_back(i, t, ngram);
            }

            for (size_t k = 1; k < max_ngram && j + k < tokens.size(); ++k) {
                ngram += " ";
                ngram += tokens[j + k];
                it = keyword_to_topics.find(ngram);
                if (it != keyword_to_topics.end()) {
                    for (const auto& t : it->second) results.emplace_back(i, t, ngram);
                }
            }
        }
    }
    return results;
}

PYBIND11_MODULE(fast_scanner, m) {
    m.def("scan_chunks", &scan_chunks, "Zero-Copy O(N) C++ text scanner");
}
