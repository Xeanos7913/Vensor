#pragma once

// Returns a vector of std::variant containing all the registered types. Useful for scripting and building new components without having to manually write the 
// enum or std::variant for the component / script container.
// Will be indispensible for script metadata serialization in the future.

#include <variant>
#include <type_traits>
#include <utility>
#include <iostream>

// --- TypeList Utilities ---
template<typename... Ts>
struct TypeList {};

template<typename T, typename List>
struct Append;

template<typename T, typename... Ts>
struct Append<T, TypeList<Ts...>> {
    using type = TypeList<Ts..., T>;
};

// --- Compile-Time Component Registry ---
template<int>
struct ComponentRegistry {
    using types = TypeList<>;
};

template<typename T>
struct RegisterComponent {
    static constexpr int id = __COUNTER__;
    using Prev = typename ComponentRegistry<id - 1>::types;
    using Current = typename Append<T, Prev>::type;

    static constexpr bool registered = ([] {
        struct Specialize : ComponentRegistry<id> {
            using types = Current;
        };
        return true;
        })();
};

// --- CRTP Auto-Register Base ---
template<typename T>
struct ComponentBase {
    static constexpr bool _auto_registered = RegisterComponent<T>::registered;
};

// --- Optional Manual Registration Macro ---
#define REGISTER_COMPONENT(Type) \
    static constexpr bool _reg_##Type = RegisterComponent<Type>::registered;

// --- Convert TypeList to std::variant ---
template<typename TList>
struct VariantFromList;

template<typename... Ts>
struct VariantFromList<TypeList<Ts...>> {
    using type = std::variant<Ts...>;
};

// --- Compile-Time ForEach ---
template<typename List> struct ForEach;

template<typename T, typename... Ts>
struct ForEach<TypeList<T, Ts...>> {
    template<typename F>
    static void apply(F&& f) {
        f.template operator() < T > ();
        ForEach<TypeList<Ts...>>::apply(std::forward<F>(f));
    }
};

template<>
struct ForEach<TypeList<>> {
    template<typename F>
    static void apply(F&&) {}
};

// --- Access Final Registered Types ---
using RegisteredComponentTypes = ComponentRegistry<__COUNTER__ - 1>::types;
using ComponentVariant = VariantFromList<RegisteredComponentTypes>::type;
